#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector, std::pair, std::array
#include <pybind11/numpy.h> // py::array_t
#include <opencv2/opencv.hpp>

#include "infer/sam3infer.hpp" 
#include "common/object.hpp"

namespace py = pybind11;

// --- 辅助函数：cv::Mat <-> numpy ---
py::array_t<uint8_t> mat_to_numpy(const cv::Mat &mat)
{
    if (mat.empty())
        return py::array_t<uint8_t>();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (mat.channels() == 1)
    {
        shape = {mat.rows, mat.cols};
        strides = {mat.step[0], mat.step[1]};
    }
    else
    {
        shape = {mat.rows, mat.cols, mat.channels()};
        strides = {mat.step[0], mat.step[1], mat.elemSize1()};
    }
    return py::array_t<uint8_t>(shape, strides, mat.data);
}

cv::Mat numpy_to_mat(py::array_t<uint8_t> &input)
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 2 && buf.ndim != 3)
    {
        throw std::runtime_error("Input numpy array must be 2D or 3D.");
    }
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    int type = (channels == 1) ? CV_8UC1 : CV_8UC3;
    return cv::Mat(rows, cols, type, buf.ptr, buf.strides[0]);
}

PYBIND11_MODULE(trtsam3, m)
{
    m.doc() = "Python bindings for Sam3Infer (One Vision, Many Prompts) using pybind11";

    // --- 1. ObjectType 枚举 ---
    py::enum_<object::ObjectType>(m, "ObjectType")
        .value("UNKNOW", object::ObjectType::UNKNOW)
        .value("DETECTION", object::ObjectType::DETECTION)
        .value("POSE", object::ObjectType::POSE)
        .value("OBB", object::ObjectType::OBB)
        .value("SEGMENTATION", object::ObjectType::SEGMENTATION)
        .export_values();

    // --- 2. 基础结果结构体 ---
    py::class_<object::Box>(m, "Box")
        .def(py::init<float, float, float, float>(),
             py::arg("left") = 0, py::arg("top") = 0, py::arg("right") = 0, py::arg("bottom") = 0)
        .def_readwrite("left", &object::Box::left)
        .def_readwrite("top", &object::Box::top)
        .def_readwrite("right", &object::Box::right)
        .def_readwrite("bottom", &object::Box::bottom)
        .def("__repr__", [](const object::Box &b)
             { return "<Box l=" + std::to_string(b.left) + ", t=" + std::to_string(b.top) +
                      ", r=" + std::to_string(b.right) + ", b=" + std::to_string(b.bottom) + ">"; });

    py::class_<object::Segmentation>(m, "Segmentation")
        .def(py::init<>())
        .def_property("mask", [](object::Segmentation &self)
                      { return mat_to_numpy(self.mask); }, [](object::Segmentation &self, py::array_t<uint8_t> array)
                      { self.mask = numpy_to_mat(array).clone(); });

    py::class_<object::DetectionBox>(m, "DetectionBox")
        .def(py::init<>())
        .def_readwrite("box", &object::DetectionBox::box)
        .def_readwrite("score", &object::DetectionBox::score)
        .def_readwrite("class_id", &object::DetectionBox::class_id)
        .def_readwrite("class_name", &object::DetectionBox::class_name)
        .def_readwrite("segmentation", &object::DetectionBox::segmentation)
        .def("__repr__", [](const object::DetectionBox &d)
             { return "<DetectionBox class='" + d.class_name + "' score=" + std::to_string(d.score) + ">"; });

    // --- 3. 新增 Sam3PromptUnit 绑定 ---
    // 这是支持单图多 Prompt 的核心单元
    py::class_<Sam3PromptUnit>(m, "Sam3PromptUnit")
        .def(py::init<>())
        .def(py::init<const std::string &, const std::vector<BoxPrompt> &>(),
             py::arg("text"), py::arg("boxes") = std::vector<BoxPrompt>())
        .def_readwrite("text", &Sam3PromptUnit::text)
        .def_readwrite("boxes", &Sam3PromptUnit::boxes)
        .def("__repr__", [](const Sam3PromptUnit &u) {
            return "<Sam3PromptUnit text='" + u.text + "' boxes_count=" + std::to_string(u.boxes.size()) + ">";
        });

    // --- 4. 修改 Sam3Input 绑定 ---
    py::class_<Sam3Input>(m, "Sam3Input")
        .def(py::init<>())
        
        // 构造函数 1: 完整模式 (Image, List[PromptUnit])
        .def(py::init([](py::array_t<uint8_t> img, const std::vector<Sam3PromptUnit> &prompts)
                      { return Sam3Input(numpy_to_mat(img), prompts); }),
             py::arg("image"), py::arg("prompts"))

        // 构造函数 2: 兼容模式 (Image, Single Text, Single Box List)
        // 方便只做一个 prompt 的时候调用，底层会自动转成 vector<Sam3PromptUnit>
        .def(py::init([](py::array_t<uint8_t> img, const std::string &txt, const std::vector<BoxPrompt> &boxes)
                      { return Sam3Input(numpy_to_mat(img), txt, boxes); }),
             py::arg("image"), py::arg("text_prompt") = "", py::arg("box_prompts") = std::vector<BoxPrompt>())

        // 属性读写
        .def_readwrite("prompts", &Sam3Input::prompts) // 暴露 prompts 列表给 Python

        // Image 特殊处理
        .def_property("image", [](Sam3Input &self)
                      { return mat_to_numpy(self.image); }, [](Sam3Input &self, py::array_t<uint8_t> array)
                      { self.image = numpy_to_mat(array).clone(); });

    // --- 5. Sam3Infer 绑定 ---
    py::class_<Sam3Infer, std::shared_ptr<Sam3Infer>>(m, "Sam3Infer")
        // 静态工厂方法
        .def_static("create_instance",
                    static_cast<std::shared_ptr<Sam3Infer> (*)(const std::string &, const std::string &, const std::string &, const std::string &, int, float)>(&Sam3Infer::create_instance),
                    py::arg("vision_path"), py::arg("text_path"), py::arg("geometry_path"), py::arg("decoder_path"),
                    py::arg("gpu_id") = 0, py::arg("confidence_threshold") = 0.5f,
                    "Create a Sam3Infer instance with all 3 encoders.")

        // 成员函数
        .def("setup_text_inputs", [](Sam3Infer &self, const std::string &text, const std::vector<int64_t> &input_ids, const std::vector<int64_t> &attention_mask)
             {
            if (input_ids.size() != 32 || attention_mask.size() != 32) throw std::runtime_error("Inputs must be 32 elements");
            std::array<int64_t, 32> arr_ids, arr_mask;
            std::copy(input_ids.begin(), input_ids.end(), arr_ids.begin());
            std::copy(attention_mask.begin(), attention_mask.end(), arr_mask.begin());
            self.setup_text_inputs(text, arr_ids, arr_mask); }, py::arg("text"), py::arg("input_ids"), py::arg("attention_mask"))
        
        .def("forwards", [](Sam3Infer &self, const std::vector<Sam3Input> &inputs, bool return_mask)
             {
            py::gil_scoped_release release;
            return self.forwards(inputs, return_mask, nullptr); }, py::arg("inputs"), py::arg("return_mask") = false);
}