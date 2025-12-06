#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 自动转换 std::vector, std::optional 等
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h> // 用于 cv::Mat 和 numpy 的转换
#include <opencv2/opencv.hpp>

#include "infer/sam3infer.hpp"
#include "common/object.hpp"

namespace py = pybind11;

// --- 辅助函数：cv::Mat -> Numpy ---
py::array_t<uint8_t> mat_to_numpy(const cv::Mat &mat)
{
    // 确保数据是连续的
    if (mat.empty())
        return py::array_t<uint8_t>();

    // 获取 buffer 信息
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

    // 创建 numpy array，并拷贝数据 (为了安全，防止 C++ 释放后 Python 崩溃)
    return py::array_t<uint8_t>(shape, strides, mat.data);
}

cv::Mat numpy_to_mat(py::array_t<uint8_t> &input)
{
    py::buffer_info buf = input.request();

    int rows, cols, channels;
    if (buf.ndim == 2)
    {
        rows = buf.shape[0];
        cols = buf.shape[1];
        channels = 1;
    }
    else if (buf.ndim == 3)
    {
        rows = buf.shape[0];
        cols = buf.shape[1];
        channels = buf.shape[2];
    }
    else
    {
        throw std::runtime_error("Input numpy array must be 2D or 3D");
    }

    int type = (channels == 1) ? CV_8UC1 : CV_8UC3;
    // 这里使用构造函数直接引用 numpy 的内存，不拷贝。
    // 注意：在 forward 函数中如果要修改图像，需要 clone
    cv::Mat mat(rows, cols, type, buf.ptr);
    return mat;
}

// --- 绑定模块 ---
PYBIND11_MODULE(trtsam3, m)
{
    m.doc() = "Python bindings for Sam3Infer using pybind11";

    py::enum_<object::ObjectType>(m, "ObjectType")
        .value("UNKNOW", object::ObjectType::UNKNOW)
        .value("DETECTION", object::ObjectType::DETECTION)
        .value("POSE", object::ObjectType::POSE)
        .value("OBB", object::ObjectType::OBB)
        .value("SEGMENTATION", object::ObjectType::SEGMENTATION)
        .export_values();

    py::class_<object::Box>(m, "Box")
        .def(py::init<float, float, float, float>(), py::arg("left") = 0, py::arg("top") = 0, py::arg("right") = 0, py::arg("bottom") = 0)
        .def_readwrite("left", &object::Box::left)
        .def_readwrite("top", &object::Box::top)
        .def_readwrite("right", &object::Box::right)
        .def_readwrite("bottom", &object::Box::bottom)
        .def("__repr__", [](const object::Box &b)
             { return "<Box l=" + std::to_string(b.left) + ", t=" + std::to_string(b.top) + ", r=" + std::to_string(b.right) + ", b=" + std::to_string(b.bottom) + ">"; });

    py::class_<object::PosePoint>(m, "PosePoint")
        .def(py::init<float, float, float>(), py::arg("x") = 0, py::arg("y") = 0, py::arg("vis") = 0)
        .def_readwrite("x", &object::PosePoint::x)
        .def_readwrite("y", &object::PosePoint::y)
        .def_readwrite("vis", &object::PosePoint::vis)
        .def("__repr__", [](const object::PosePoint &p)
             { return "<PosePoint x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y) + ", vis=" + std::to_string(p.vis) + ">"; });

    py::class_<object::Pose>(m, "Pose")
        .def(py::init<>())
        .def_readwrite("points", &object::Pose::points);

    py::class_<object::Obb>(m, "Obb")
        .def(py::init<float, float, float, float, float>(), py::arg("cx") = 0, py::arg("cy") = 0, py::arg("width") = 0, py::arg("height") = 0, py::arg("angle") = 0)
        .def_readwrite("cx", &object::Obb::cx)
        .def_readwrite("cy", &object::Obb::cy)
        .def_readwrite("width", &object::Obb::w)
        .def_readwrite("height", &object::Obb::h)
        .def_readwrite("angle", &object::Obb::angle);

    py::class_<object::Segmentation>(m, "Segmentation")
        .def(py::init<>())
        .def_property("mask",
                      // Getter: C++ -> Python (cv::Mat -> numpy)
                      [](object::Segmentation &self)
                      { return mat_to_numpy(self.mask); },
                      // Setter: Python -> C++ (numpy -> cv::Mat)
                      [](object::Segmentation &self, py::array_t<uint8_t> array)
                      {
                // 注意：为了安全，建议 clone 一份数据，
                // 因为 numpy_to_mat 只是引用了 numpy 的内存
                self.mask = numpy_to_mat(array).clone(); })
        // 绑定其他成员函数 (如果有的话)
        .def("keep_largest_part", &object::Segmentation::keep_largest_part)
        // .def("align_to_left_top", &object::Segmentation::align_to_left_top)
        ;

    // 绑定核心的 DetectionBox 结构
    py::class_<object::DetectionBox>(m, "DetectionBox")
        .def(py::init<>())
        .def_readwrite("type", &object::DetectionBox::type)
        .def_readwrite("box", &object::DetectionBox::box)
        .def_readwrite("score", &object::DetectionBox::score)
        .def_readwrite("class_id", &object::DetectionBox::class_id)
        .def_readwrite("class_name", &object::DetectionBox::class_name)
        .def_readwrite("pose", &object::DetectionBox::pose)
        .def_readwrite("obb", &object::DetectionBox::obb)
        .def_readwrite("segmentation", &object::DetectionBox::segmentation)
        .def("__repr__", [](const object::DetectionBox &d)
             { return "<DetectionBox class='" + d.class_name + "' score=" + std::to_string(d.score) + ">"; });

    py::bind_vector<std::vector<object::DetectionBox>>(m, "DetectionBoxArray");

    py::class_<Sam3Infer>(m, "Sam3Infer")
        .def(py::init<const std::string &, const std::string &, const std::string &, int, float>(),
             py::arg("vision_encoder_path"),
             py::arg("text_encoder_path"),
             py::arg("decoder_path"),
             py::arg("gpu_id") = 0,
             py::arg("confidence_threshold") = 0.5f)

        .def("load_engines", &Sam3Infer::load_engines)

        .def("setup_text_inputs", [](Sam3Infer &self, const std::string &text, const std::vector<int64_t> &input_ids, const std::vector<int64_t> &attention_mask)
             {
                if (input_ids.size() != 32 || attention_mask.size() != 32) 
                {
                    throw std::runtime_error("input_ids and attention_mask must be exactly 32 elements");
                }
            std::array<int64_t, 32> arr_ids;
            std::array<int64_t, 32> arr_mask;
            std::copy(input_ids.begin(), input_ids.end(), arr_ids.begin());
            std::copy(attention_mask.begin(), attention_mask.end(), arr_mask.begin());
            
            self.setup_text_inputs(text, arr_ids, arr_mask); }, py::arg("text"), py::arg("input_ids"), py::arg("attention_mask"))

        .def("forward", [](Sam3Infer &self, py::array_t<uint8_t> input_image, const std::string &input_text)
             {
            cv::Mat img = numpy_to_mat(input_image);
            return self.forward(img, input_text, nullptr); }, py::arg("input_image"), py::arg("input_text"));
}