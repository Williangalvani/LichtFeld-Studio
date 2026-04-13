/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <Python.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/logger.hpp"
#include "python/gil.hpp"
#include "python/runner.hpp"
#include "training/control/command_api.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
    std::filesystem::path findPythonModuleDir();
    void prependPythonPath(const std::filesystem::path& path);
} // namespace

class PythonIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto module_dir = findPythonModuleDir();
        ASSERT_FALSE(module_dir.empty()) << "Could not locate built lichtfeld module for Python tests";
        prependPythonPath(module_dir);
        lfs::python::ensure_initialized();
    }

    std::filesystem::path createTempScript(const std::string& content) {
        auto temp_dir = std::filesystem::temp_directory_path();
        auto script_path = temp_dir / "test_script.py";
        std::ofstream ofs(script_path);
        ofs << content;
        ofs.close();
        return script_path;
    }
};

namespace {
    struct PythonTensorResult {
        std::vector<int64_t> shape;
        std::vector<float> values;
    };

    bool containsLichtfeldModule(const std::filesystem::path& dir) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec)) {
            return false;
        }

        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec) {
                continue;
            }

            const auto filename = it->path().filename().string();
            const auto ext = it->path().extension().string();
            if ((ext == ".so" || ext == ".pyd") && filename.rfind("lichtfeld", 0) == 0) {
                return true;
            }
        }

        return false;
    }

    std::filesystem::path findPythonModuleDir() {
        std::error_code ec;
        const auto cwd = std::filesystem::current_path(ec);
        const auto project_root = std::filesystem::path(PROJECT_ROOT_PATH);

        for (const auto& candidate : {
                 cwd / "src" / "python",
                 cwd.parent_path() / "src" / "python",
                 project_root / "build" / "src" / "python",
             }) {
            if (containsLichtfeldModule(candidate)) {
                return candidate;
            }
        }

        return {};
    }

    void prependPythonPath(const std::filesystem::path& path) {
        const auto value = path.string();
        const char* existing = std::getenv("PYTHONPATH");
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        const std::string combined =
            existing && *existing ? value + separator + std::string(existing) : value;

#ifdef _WIN32
        _putenv_s("PYTHONPATH", combined.c_str());
#else
        setenv("PYTHONPATH", combined.c_str(), 1);
#endif
    }

    std::string consumePythonError() {
        if (!PyErr_Occurred()) {
            return "unknown Python error";
        }

        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);

        std::string message = "unknown Python error";
        if (value) {
            PyObject* as_str = PyObject_Str(value);
            if (as_str) {
                if (const char* utf8 = PyUnicode_AsUTF8(as_str)) {
                    message = utf8;
                }
                Py_DECREF(as_str);
            }
        }

        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return message;
    }

    PythonTensorResult runPythonTensorSnippet(const std::string& script) {
        const lfs::python::GilAcquire gil;

        PyObject* globals = PyDict_New();
        PyObject* locals = PyDict_New();
        if (!globals || !locals) {
            Py_XDECREF(globals);
            Py_XDECREF(locals);
            throw std::runtime_error("Failed to allocate Python dictionaries");
        }

        PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());

        PyObject* exec_result = PyRun_String(script.c_str(), Py_file_input, globals, locals);
        if (!exec_result) {
            const auto error = consumePythonError();
            Py_DECREF(globals);
            Py_DECREF(locals);
            throw std::runtime_error(error);
        }
        Py_DECREF(exec_result);

        auto* const shape_obj = PyDict_GetItemString(locals, "result_shape");
        auto* const values_obj = PyDict_GetItemString(locals, "result_values");
        if (!shape_obj || !PyTuple_Check(shape_obj) || !values_obj || !PyList_Check(values_obj)) {
            Py_DECREF(globals);
            Py_DECREF(locals);
            throw std::runtime_error("Python snippet did not populate result_shape/result_values");
        }

        PythonTensorResult result;
        result.shape.reserve(static_cast<size_t>(PyTuple_Size(shape_obj)));
        for (Py_ssize_t i = 0; i < PyTuple_Size(shape_obj); ++i) {
            result.shape.push_back(PyLong_AsLongLong(PyTuple_GetItem(shape_obj, i)));
        }

        result.values.reserve(static_cast<size_t>(PyList_Size(values_obj)));
        for (Py_ssize_t i = 0; i < PyList_Size(values_obj); ++i) {
            result.values.push_back(static_cast<float>(PyFloat_AsDouble(PyList_GetItem(values_obj, i))));
        }

        Py_DECREF(globals);
        Py_DECREF(locals);
        return result;
    }

    std::vector<long long> runPythonHookContextSnippet(const std::string& registration_script,
                                                       const lfs::training::HookContext& snapshot_ctx,
                                                       const lfs::training::HookContext& callback_ctx) {
        lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
        lfs::training::ControlBoundary::instance().clear_all();
        lfs::training::CommandCenter::instance().update_snapshot(
            snapshot_ctx,
            /*max_iterations=*/5000,
            /*is_paused=*/false,
            /*is_running=*/true,
            /*stop_requested=*/false,
            lfs::training::TrainingPhase::SafeControl);

        PyObject* globals = nullptr;
        {
            const lfs::python::GilAcquire gil;
            globals = PyDict_New();
            if (!globals) {
                throw std::runtime_error("Failed to allocate Python globals");
            }

            PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
            PyObject* exec_result = PyRun_String(registration_script.c_str(), Py_file_input, globals, globals);
            if (!exec_result) {
                const auto error = consumePythonError();
                Py_DECREF(globals);
                throw std::runtime_error(error);
            }
            Py_DECREF(exec_result);
        }

        lfs::training::ControlBoundary::instance().notify(
            lfs::training::ControlHook::PostStep,
            callback_ctx);
        lfs::training::ControlBoundary::instance().drain_callbacks();

        std::vector<long long> result;
        {
            const lfs::python::GilAcquire gil;
            auto* records_obj = PyDict_GetItemString(globals, "records");
            if (!records_obj || !PyList_Check(records_obj) || PyList_Size(records_obj) != 1) {
                lfs::training::ControlBoundary::instance().clear_all();
                Py_DECREF(globals);
                throw std::runtime_error("Hook script did not record exactly one callback invocation");
            }

            auto* record = PyList_GetItem(records_obj, 0);
            if (!record || !PyTuple_Check(record)) {
                lfs::training::ControlBoundary::instance().clear_all();
                Py_DECREF(globals);
                throw std::runtime_error("Recorded hook result is not a tuple");
            }

            result.reserve(static_cast<size_t>(PyTuple_Size(record)));
            for (Py_ssize_t i = 0; i < PyTuple_Size(record); ++i) {
                result.push_back(PyLong_AsLongLong(PyTuple_GetItem(record, i)));
            }

            // Drop retained Python callbacks before releasing the globals dict.
            lfs::training::ControlBoundary::instance().clear_all();
            Py_DECREF(globals);
        }
        return result;
    }

    void comparePythonResultToTorch(const PythonTensorResult& custom,
                                    const torch::Tensor& reference,
                                    const std::string& context,
                                    const float rtol = 1e-5f,
                                    const float atol = 1e-7f) {
        const auto reference_cpu = reference.cpu().contiguous();

        ASSERT_EQ(custom.shape.size(), static_cast<size_t>(reference_cpu.dim())) << context << ": rank mismatch";
        for (size_t i = 0; i < custom.shape.size(); ++i) {
            ASSERT_EQ(custom.shape[i], reference_cpu.size(i))
                << context << ": shape mismatch at dim " << i;
        }

        const auto* const reference_values = reference_cpu.data_ptr<float>();
        ASSERT_EQ(custom.values.size(), static_cast<size_t>(reference_cpu.numel())) << context << ": numel mismatch";

        for (size_t i = 0; i < custom.values.size(); ++i) {
            const float diff = std::abs(custom.values[i] - reference_values[i]);
            const float threshold = atol + rtol * std::abs(reference_values[i]);
            EXPECT_LE(diff, threshold) << context << ": mismatch at index " << i;
        }
    }

    bool formatterUnavailable(const lfs::python::FormatResult& result) {
        return !result.success &&
               (result.error.find("uv not found") != std::string::npos ||
                result.error.find("Failed to create venv for black") != std::string::npos ||
                result.error.find("ImportError") != std::string::npos);
    }
} // namespace

TEST_F(PythonIntegrationTest, InitializationSucceeds) {
    // Just verify that initialization doesn't throw
    EXPECT_NO_THROW(lfs::python::ensure_initialized());
}

TEST_F(PythonIntegrationTest, OutputCallbackCanBeSet) {
    bool callback_set = false;
    lfs::python::set_output_callback([&](const std::string&, bool) { callback_set = true; });
    EXPECT_TRUE(true); // If we got here, setting the callback didn't crash
}

TEST_F(PythonIntegrationTest, OutputRedirectCanBeInstalled) {
    // This should not throw
    EXPECT_NO_THROW(lfs::python::install_output_redirect());
}

TEST_F(PythonIntegrationTest, EmptyScriptListSucceeds) {
    auto result = lfs::python::run_scripts({});
    EXPECT_TRUE(result.has_value()) << "Empty script list should succeed";
}

TEST_F(PythonIntegrationTest, FormatPythonCodePreservesValidBlockIndentation) {
    const auto result = lfs::python::format_python_code("if True:\n    print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "if True:\n    print(\"x\")\n");
}

TEST_F(PythonIntegrationTest, FormatPythonCodeDedentsIndentedSnippet) {
    const auto result = lfs::python::format_python_code("    if True:\n        print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "if True:\n    print(\"x\")\n");
}

TEST_F(PythonIntegrationTest, FormatPythonCodeRepairsUnexpectedTopLevelIndent) {
    const auto result = lfs::python::format_python_code(
        "import lichtfeld as lf\n    scene = lf.get_scene()\nprint('hello world')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code.find("\n    scene = lf.get_scene()"), std::string::npos);
    EXPECT_NE(result.code.find("scene = lf.get_scene()"), std::string::npos);
    EXPECT_NE(result.code.find("print(\"hello world\")"), std::string::npos);
}

TEST_F(PythonIntegrationTest, FormatPythonCodeCommentsLeadingPreambleBullets) {
    const auto result = lfs::python::format_python_code(
        "1. SOURCE_NAME if set\n"
        "2. currently selected node\n"
        "3. first splat node in the scene\n"
        "\n"
        "from pathlib import Path\n"
        "import lichtfeld as lf\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_NE(result.code.find("# 1. SOURCE_NAME if set"), std::string::npos);
    EXPECT_NE(result.code.find("# 2. currently selected node"), std::string::npos);
    EXPECT_NE(result.code.find("from pathlib import Path"), std::string::npos);
    EXPECT_NE(result.code.find("import lichtfeld as lf"), std::string::npos);
}

TEST_F(PythonIntegrationTest, FormatPythonCodeReportsSyntaxErrorWithoutUnexpectedResultFallback) {
    const auto result = lfs::python::format_python_code("import os\nif True print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
    EXPECT_EQ(result.error.find("unexpected result"), std::string::npos);
}

TEST_F(PythonIntegrationTest, PyTensorBooleanRowMaskIndexingMatchesTorch) {
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
t = lf.Tensor.arange(1, 13, 1, device="cpu", dtype="float32").reshape([4, 3])
mask = lf.Tensor.arange(0, 4, 1, device="cpu", dtype="float32") != 1
selected = t[mask]
result_shape = tuple(selected.shape)
result_values = selected.flatten().tolist()
)PY");

    const auto torch_tensor = torch::arange(1, 13, torch::kFloat32).reshape({4, 3});
    const auto torch_mask = torch::tensor(std::vector<int>{1, 0, 1, 1}, torch::kInt32).to(torch::kBool);
    const auto torch_result = torch_tensor.index({torch_mask});

    comparePythonResultToTorch(result, torch_result, "PyTensor row mask");
}

TEST_F(PythonIntegrationTest, PyTensorElementwiseBooleanMaskIndexingMatchesTorch) {
    const auto result = runPythonTensorSnippet(R"PY(
import lichtfeld as lf
t = lf.Tensor.arange(1, 13, 1, device="cpu", dtype="float32").reshape([4, 3])
mask = (t == 1) | (t == 4) | (t == 6) | (t == 9) | (t == 10)
selected = t[mask]
result_shape = tuple(selected.shape)
result_values = selected.tolist()
)PY");

    const auto torch_tensor = torch::arange(1, 13, torch::kFloat32).reshape({4, 3});
    const auto torch_mask =
        (torch_tensor == 1) |
        (torch_tensor == 4) |
        (torch_tensor == 6) |
        (torch_tensor == 9) |
        (torch_tensor == 10);
    const auto torch_result = torch_tensor.masked_select(torch_mask);

    comparePythonResultToTorch(result, torch_result, "PyTensor elementwise mask");
}

TEST_F(PythonIntegrationTest, DecoratorHookContextUsesLiveHookSnapshot) {
    const lfs::training::HookContext stale_snapshot{
        .iteration = 0,
        .loss = 0.0f,
        .num_gaussians = 0,
        .is_refining = false,
        .trainer = nullptr,
    };
    const lfs::training::HookContext live_callback{
        .iteration = 1047,
        .loss = 0.125f,
        .num_gaussians = 98765,
        .is_refining = true,
        .trainer = nullptr,
    };

    const auto result = runPythonHookContextSnippet(
        R"PY(
	import lichtfeld as lf
	records = []
	
	@lf.on_post_step
	def _hook(hook):
	    ctx = lf.context()
	    records.append((
	        hook["iter"],
	        hook["iteration"],
	        hook["num_splats"],
	        hook["num_gaussians"],
	        ctx.iteration,
	        ctx.num_gaussians,
	    ))
	)PY",
        stale_snapshot,
        live_callback);

    ASSERT_EQ(result.size(), 6u);
    EXPECT_EQ(result[0], live_callback.iteration);
    EXPECT_EQ(result[1], live_callback.iteration);
    EXPECT_EQ(result[2], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[3], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[4], live_callback.iteration);
    EXPECT_EQ(result[5], static_cast<long long>(live_callback.num_gaussians));
}

TEST_F(PythonIntegrationTest, ScopedHandlerHookContextUsesLiveHookSnapshot) {
    const lfs::training::HookContext stale_snapshot{
        .iteration = 0,
        .loss = 0.0f,
        .num_gaussians = 0,
        .is_refining = false,
        .trainer = nullptr,
    };
    const lfs::training::HookContext live_callback{
        .iteration = 1008,
        .loss = 0.25f,
        .num_gaussians = 54321,
        .is_refining = false,
        .trainer = nullptr,
    };

    const auto result = runPythonHookContextSnippet(
        R"PY(
	import lichtfeld as lf
	records = []
	handler = lf.ScopedHandler()
	
	def _hook(hook):
	    ctx = lf.context()
	    records.append((
	        hook["iter"],
	        hook["iteration"],
	        hook["num_splats"],
	        hook["num_gaussians"],
	        ctx.iteration,
	        ctx.num_gaussians,
	    ))
	
	handler.on_post_step(_hook)
	)PY",
        stale_snapshot,
        live_callback);

    ASSERT_EQ(result.size(), 6u);
    EXPECT_EQ(result[0], live_callback.iteration);
    EXPECT_EQ(result[1], live_callback.iteration);
    EXPECT_EQ(result[2], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[3], static_cast<long long>(live_callback.num_gaussians));
    EXPECT_EQ(result[4], live_callback.iteration);
    EXPECT_EQ(result[5], static_cast<long long>(live_callback.num_gaussians));
}

// NOTE: Tests that actually execute Python scripts require the lichtfeld module
// to be importable, which depends on the CommandCenter and training infrastructure.
// These are better tested via integration tests (running training with --python-script).
