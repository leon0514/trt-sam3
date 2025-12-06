#include "common/createObject.hpp"
#include "common/object.hpp"
#include <tuple>

namespace
{

    std::tuple<float, float, float, float> getAABBFromObb(float cx, float cy, float w, float h, float angle_degrees)
    {
        float cos_a = std::cos(angle_degrees);
        float sin_a = std::sin(angle_degrees);

        float x1 = cx - w / 2 * cos_a + h / 2 * sin_a;
        float y1 = cy - w / 2 * sin_a - h / 2 * cos_a;
        float x2 = cx + w / 2 * cos_a + h / 2 * sin_a;
        float y2 = cy + w / 2 * sin_a - h / 2 * cos_a;
        float x3 = cx + w / 2 * cos_a - h / 2 * sin_a;
        float y3 = cy + w / 2 * sin_a + h / 2 * cos_a;
        float x4 = cx - w / 2 * cos_a - h / 2 * sin_a;
        float y4 = cy - w / 2 * sin_a + h / 2 * cos_a;

        float left = std::min({x1, x2, x3, x4});
        float top = std::min({y1, y2, y3, y4});
        float right = std::max({x1, x2, x3, x4});
        float bottom = std::max({y1, y2, y3, y4});

        return std::make_tuple(left, top, right, bottom);
    }

}

namespace object
{
    DetectionBox createBox(float left, float top, float right, float bottom, float score, int class_id, const std::string &class_name)
    {
        DetectionBox box;
        box.type = ObjectType::DETECTION;
        box.box = Box(left, top, right, bottom);
        box.score = score;
        box.class_id = class_id;
        box.class_name = class_name;
        return box;
    }

    DetectionBox createPositionBox(float left, float top, float right, float bottom, float score, int class_id, const std::string &class_name)
    {
        DetectionBox box;
        box.type = ObjectType::POSITION;
        box.box = Box(left, top, right, bottom);
        box.score = score;
        box.class_id = class_id;
        box.class_name = class_name;
        return box;
    }

    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name)
    {
        DetectionBox box;
        box.type = ObjectType::TRACK;
        box.box = Box(left, top, right, bottom);
        box.score = score;
        box.class_name = class_name;
        box.track.emplace();
        box.track->track_id = track_id;
        return box;
    }

    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name, const object::Pose &pose)
    {
        DetectionBox box;
        box.type = ObjectType::TRACK;
        box.box = Box(left, top, right, bottom);
        box.score = score;
        box.class_name = class_name;
        box.track.emplace();
        box.track->track_id = track_id;
        box.pose.emplace();
        box.pose = pose;
        return box;
    }

    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name, const object::Obb &obb)
    {
        DetectionBox box;
        box.type = ObjectType::TRACK;
        box.box = Box(left, top, right, bottom);
        box.score = score;
        box.class_name = class_name;
        box.track.emplace();
        box.track->track_id = track_id;
        box.obb.emplace();
        box.obb = obb;
        return box;
    }

    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name, const object::Segmentation &seg)
    {
        DetectionBox box;
        box.type = ObjectType::TRACK;
        box.box = Box(left, top, right, bottom);
        box.score = score;
        box.class_name = class_name;
        box.track.emplace();
        box.track->track_id = track_id;
        box.segmentation.emplace();
        box.segmentation = seg;
        return box;
    }

    DetectionBox createObbBox(float cx, float cy, float w, float h, float angle, float score, int class_id, const std::string &class_name)
    {
        DetectionBox box;
        box.type = ObjectType::OBB;
        box.obb = Obb(cx, cy, w, h, angle);
        box.score = score;
        box.class_id = class_id;
        box.class_name = class_name;

        std::tuple<float, float, float, float> aabb = getAABBFromObb(cx, cy, w, h, angle);
        box.box = Box(std::get<0>(aabb), std::get<1>(aabb), std::get<2>(aabb), std::get<3>(aabb));
        return box;
    }

    DetectionBox createPoseBox(float left, float top, float right, float bottom, const std::vector<PosePoint> &pose_points, float score, int class_id, const std::string &class_name)
    {
        DetectionBox box;
        box.type = ObjectType::POSE;
        box.box = Box(left, top, right, bottom);
        box.pose.emplace();
        box.pose->points = pose_points;
        box.score = score;
        box.class_id = class_id;
        box.class_name = class_name;
        return box;
    }

    DetectionBox createSegmentationBox(float left, float top, float right, float bottom, const cv::Mat &mask, float score, int class_id, const std::string &class_name)
    {
        DetectionBox box;
        box.type = ObjectType::SEGMENTATION;
        box.box = Box(left, top, right, bottom);
        box.segmentation.emplace();
        box.segmentation->mask = mask.clone();
        box.score = score;
        box.class_id = class_id;
        box.class_name = class_name;
        return box;
    }

    DetectionBox createDepthProBox(const cv::Mat &depth, float fog_data)
    {
        DetectionBox box;
        box.type = ObjectType::DEPTH_PRO;
        box.box = Box(0, 0, depth.cols, depth.rows);
        box.depth.emplace();
        box.depth->depth = depth.clone();
        box.depth->fog_data = fog_data;
        box.score = 1.0;
        box.class_id = 0;
        box.class_name = "depth";
        return box;
    }

    DetectionBox createDepthAnythingBox(const cv::Mat &depth)
    {
        DetectionBox box;
        box.type = ObjectType::DEPTH_ANYTHING;
        box.box = Box(0, 0, depth.cols, depth.rows);
        box.depth.emplace();
        box.depth->depth = depth.clone();
        box.score = 1.0;
        box.class_id = 0;
        box.class_name = "depth";
        return box;
    }

}