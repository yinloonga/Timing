//
// Created by tommy on 4/25/19.
//

#ifndef ESDF_TOOLS_INCLUDE_FIESTA_H_
#define ESDF_TOOLS_INCLUDE_FIESTA_H_
#include "ESDFMap.h"
#include "raycast.h"
#include "timing.h"
#include <iostream>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <thread>
#include <type_traits>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud.h>
#include <unordered_set>
#include <visualization_msgs/Marker.h>
#include <fstream>
#include <vector>
namespace fiesta {

// sensor_msgs::PointCloud2::ConstPtr
// sensor_msgs::Image::ConstPtr

// geometry_msgs::PoseStamped::ConstPtr
// nav_msgs::Odometry::ConstPtr
// geometry_msgs::TransformStamped::ConstPtr
template<class DepthMsgType, class PoseMsgType>
class Fiesta {
private:
    Parameters parameters_;
    ESDFMap *esdf_map_;
#ifdef SIGNED_NEEDED
    ESDFMap *inv_esdf_map_;
#endif
    bool new_msg_ = false;
    pcl::PointCloud<pcl::PointXYZ> cloud_;
#ifndef PROBABILISTIC
    sensor_msgs::PointCloud2::ConstPtr sync_pc_;
#endif
    ros::Publisher slice_pub_, occupancy_pub_, text_pub_;
    ros::Subscriber transform_sub_, depth_sub_, path_sub_;
    ros::Timer update_mesh_timer_;
    Eigen::Vector3d sync_pos_, raycast_origin_, cur_pos_;
    Eigen::Quaterniond sync_q_;

    std::queue<std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>> transform_queue_;
    std::queue<DepthMsgType> depth_queue_;
    DepthMsgType sync_depth_;

    cv::Mat img_[2];
    Eigen::Matrix4d transform_, last_transform_;
    uint image_cnt_ = 0, esdf_cnt_ = 0, tot_ = 0;
#ifdef HASH_TABLE
    std::unordered_set<int> set_free_, set_occ_;
#else
    std::vector<int> set_free_, set_occ_;
#endif
    void Visualization(ESDFMap *esdf_map, bool global_vis, const std::string &text);
#ifdef PROBABILISTIC
    void RaycastProcess(int i, int part, int tt);
    void RaycastMultithread();
#endif

    double GetInterpolation(const cv::Mat &img, double u, double v);

    void DepthConversion();

    void SynchronizationAndProcess();

    void DepthCallback(const DepthMsgType &depth_map);

    void PoseCallback(const PoseMsgType &msg);
    void PathCallback(const nav_msgs::Path::ConstPtr &path);
    Eigen::Vector3d path_pos_;
    Eigen::Vector4d path_q_;

    void UpdateEsdfEvent(const ros::TimerEvent & /*event*/);
public:
    Fiesta(ros::NodeHandle node);
    ~Fiesta();
  ros::Publisher robot_model_pub_;
  void publishRobotMesh();
};

template<class DepthMsgType, class PoseMsgType>
Fiesta<DepthMsgType, PoseMsgType>::Fiesta(ros::NodeHandle node) {
     parameters_.SetParameters(node);
#ifdef HASH_TABLE
     esdf_map_ = new ESDFMap(Eigen::Vector3d(0, 0, 0), parameters_.resolution_/* 0.05 */, parameters_.reserved_size_);
#ifdef SIGNED_NEEDED
       inv_esdf_map_ = new ESDFMap(Eigen::Vector3d(0, 0, 0), parameters_.resolution_, parameters_.reserved_size_);
#endif
#else
     // l_cornor_=[-20.f, -20.f, -5.f] resolution_=0.05  map_size_=[40, 40, 10]
     esdf_map_ = new ESDFMap(parameters_.l_cornor_, parameters_.resolution_, parameters_.map_size_,parameters_.slice_x_,parameters_.slice_y_);
#ifdef SIGNED_NEEDED
     inv_esdf_map_ = new ESDFMap(parameters_.l_cornor_, parameters_.resolution_, parameters_.map_size_);
#endif
#endif

#ifdef PROBABILISTIC
     // p_hit=0.7 p_miss=0.35 p_min=0.12 p_max=0.97  p_occ=0.80
     esdf_map_->SetParameters(parameters_.p_hit_, parameters_.p_miss_,
                              parameters_.p_min_, parameters_.p_max_, parameters_.p_occ_);
#endif
#ifndef HASH_TABLE
     set_free_.resize(esdf_map_->grid_total_size_);
     set_occ_.resize(esdf_map_->grid_total_size_);
     std::fill(set_free_.begin(), set_free_.end(), 0);
     std::fill(set_occ_.begin(), set_occ_.end(), 0);
#endif
     // For Jie Bao
//     transform_sub_ = node.subscribe("/vins_estimator/camera_pose", 10, &Fiesta::PoseCallback, this);
//     depth_sub_ = node.subscribe("/camera/depth/image_rect_raw", 10, &Fiesta::DepthCallback, this);
    transform_sub_ = node.subscribe("transform", 10, &Fiesta::PoseCallback, this);
    depth_sub_ = node.subscribe("depth", 10, &Fiesta::DepthCallback, this);
    path_sub_ = node.subscribe("path", 10, &Fiesta::PathCallback, this);

     // Cow_and_Lady
     // depth_sub_ = node.subscribe("/camera/depth_registered/points", 1000, PointcloudCallback);
     // transform_sub_ = node.subscribe("/kinect/vrpn_client/estimated_transform", 1000, PoseCallback);

     //EuRoC
//    depth_sub_ = node.subscribe("/dense_stereo/pointcloud", 1000, PointcloudCallback);
//    transform_sub_ = node.subscribe("/vicon/firefly_sbx/firefly_sbx", 1000, PoseCallback);

     // tag slice_pub发布者名称
     slice_pub_ = node.advertise<visualization_msgs::Marker>("ESDFMap/slice", 1, true);
     occupancy_pub_ = node.advertise<sensor_msgs::PointCloud>("ESDFMap/occ_pc", 1, true);
     text_pub_ = node.advertise<visualization_msgs::Marker>("ESDFMap/text", 1, true);
     robot_model_pub_ =
      node.advertise<visualization_msgs::Marker>("Robot_model", 100);

     update_mesh_timer_ =
         node.createTimer(ros::Duration(parameters_.update_esdf_every_n_sec_/*0.1*/),
                          &Fiesta::UpdateEsdfEvent, this);
}

template<class DepthMsgType, class PoseMsgType>
Fiesta<DepthMsgType, PoseMsgType>::~Fiesta() {
     delete esdf_map_;
#ifdef SIGNED_NEEDED
     delete inv_esdf_map_;
#endif
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::Visualization(ESDFMap *esdf_map, bool global_vis, const std::string &text) {
     if (esdf_map!=nullptr) {
          std::cout << "Visualization" << std::endl;
          if (global_vis)
               esdf_map->SetOriginalRange();
          else
               esdf_map->SetUpdateRange(cur_pos_/* cur_pos_=sync_pos_ */ - parameters_.radius_/* [3.f, 3.f, 1.5f] */, cur_pos_ + parameters_.radius_, false);

          sensor_msgs::PointCloud pc;
          esdf_map->GetPointCloud(pc, parameters_.vis_lower_bound_, parameters_.vis_upper_bound_);
          occupancy_pub_.publish(pc);

          visualization_msgs::Marker slice_marker;
          esdf_map->GetSliceMarker(slice_marker, parameters_.slice_vis_level_, 100,
                                   Eigen::Vector4d(0, 1.0, 0, 1), parameters_.slice_vis_max_dist_,cur_pos_);
          slice_pub_.publish(slice_marker);
          // visualization drone
          // publishRobotMesh();
     }
     if (!text.empty()) {
          visualization_msgs::Marker marker;
          marker.header.frame_id = "world";
          marker.header.stamp = ros::Time::now();
          marker.id = 3456;
          marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
          marker.action = visualization_msgs::Marker::MODIFY;

          marker.pose.position.x = 8.0;
          marker.pose.position.y = 2.0;
          marker.pose.position.z = 3.0;
          marker.pose.orientation.x = 0.0;
          marker.pose.orientation.y = 0.0;
          marker.pose.orientation.z = 0.0;
          marker.pose.orientation.w = 1.0;

          marker.text = text;

          marker.scale.x = 0.3;
          marker.scale.y = 0.3;
          marker.scale.z = 0.6;

          marker.color.r = 0.0f;
          marker.color.g = 0.0f;
          marker.color.b = 1.0f;
          marker.color.a = 1.0f;
          text_pub_.publish(marker);
     }
}

#ifdef PROBABILISTIC

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::RaycastProcess(int i, int part, int tt) {
     Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
     // 处理每一帧中的点云数据
     for (int idx = part*i; idx < part*(i + 1); idx++) {
          std::vector<Eigen::Vector3d> output;
          if (idx > cloud_.points.size())
               break;
          pcl::PointXYZ pt = cloud_.points[idx];
          int cnt = 0;
          if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
               continue;
          Eigen::Vector4d tmp = transform_*Eigen::Vector4d(pt.x, pt.y, pt.z, 1);  // 齐次坐标系
          Eigen::Vector3d point = Eigen::Vector3d(tmp(0), tmp(1), tmp(2))/tmp(3);  // 非齐次坐标系

          int tmp_idx;
          double length = (point - raycast_origin_).norm();  // 对于vector, norm()返回的是向量的模(向量的二范数)
          if (length < parameters_.min_ray_length_/*0.5*/)
               continue;
          else if (length > parameters_.max_ray_length_)/*5.0*/ {
               point = (point - raycast_origin_)/length*parameters_.max_ray_length_ + raycast_origin_;
               tmp_idx = esdf_map_->SetOccupancy((Eigen::Vector3d) point, 0);
          } else
               tmp_idx = esdf_map_->SetOccupancy((Eigen::Vector3d) point, 1);
#ifdef SIGNED_NEEDED
          tmp_idx = inv_esdf_map_->SetOccupancy((Eigen::Vector3d) point, 0);
#endif
//         //TODO: -10000 ?

          if (tmp_idx!=-10000) {
#ifdef HASH_TABLE
               if (set_occ_.find(tmp_idx) != set_occ_.end())
                   continue;
                 else set_occ_.insert(tmp_idx);
#else
               if (set_occ_[tmp_idx]==tt)
                    continue;
               else
                    set_occ_[tmp_idx] = tt;
#endif
          }
          Raycast(raycast_origin_/parameters_.resolution_/*start*/,
                  point/parameters_.resolution_/*end*/,
                  parameters_.l_cornor_/parameters_.resolution_/*min*/,
                  parameters_.r_cornor_/parameters_.resolution_/*max*/,
                  &output/*output*/);

          for (int i = output.size() - 2; i >= 0; i--) {
               Eigen::Vector3d tmp = (output[i] + half)*parameters_.resolution_;

               length = (tmp - raycast_origin_).norm();
               if (length < parameters_.min_ray_length_)
                    break;
               if (length > parameters_.max_ray_length_)
                    continue;
               int tmp_idx;
               tmp_idx = esdf_map_->SetOccupancy(tmp, 0);
#ifdef SIGNED_NEEDED
               tmp_idx = inv_esdf_map_->SetOccupancy(tmp, 1);
#endif
               //TODO: -10000 ?
               if (tmp_idx!=-10000) {
#ifdef HASH_TABLE
                    if (set_free_.find(tmp_idx) != set_free_.end()) {
                        if (++cnt >= 1) {
                          cnt = 0;
                          break;
                        }
                      } else {
                        set_free_.insert(tmp_idx);
                        cnt = 0;
                      }
#else
                    if (set_free_[tmp_idx]==tt) {
                         if (++cnt >= 1) {
                              cnt = 0;
                              break;
                         }
                    } else {
                         set_free_[tmp_idx] = tt;
                         cnt = 0;
                    }
#endif
               }
          }
     }
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::RaycastMultithread() {
     // TODO: when using vector, this is not needed
#ifdef HASH_TABLE
     set_free_.clear();
       set_occ_.clear();
#endif
     int tt = ++tot_;
     timing::Timer raycastingTimer("raycasting");

     if (parameters_.ray_cast_num_thread_==0) {
          RaycastProcess(0, cloud_.points.size(), tt);
     } else {
          int part = cloud_.points.size()/parameters_.ray_cast_num_thread_;
          std::list<std::thread> integration_threads = std::list<std::thread>();
          for (size_t i = 0; i < parameters_.ray_cast_num_thread_; ++i) {
               integration_threads.emplace_back(&Fiesta::RaycastProcess, this, i, part, tt);
          }
          for (std::thread &thread : integration_threads) {
               thread.join();
          }
     }
     raycastingTimer.Stop();
}

#endif // PROBABILISTIC

template<class DepthMsgType, class PoseMsgType>
double Fiesta<DepthMsgType, PoseMsgType>::GetInterpolation(const cv::Mat &img, double u, double v) {
     int vu = img.at<uint16_t>(v, u);
     int v1u = img.at<uint16_t>(v + 1, u);
     int vu1 = img.at<uint16_t>(v, u + 1);
     int v1u1 = img.at<uint16_t>(v + 1, u + 1);
     float a = u - (float) u;
     float c = v - (float) v;
     return (vu*(1.f - a) + vu1*a)*(1.f - c) + (v1u*(1.f - a) + v1u1*a)*c;
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::DepthConversion() {
     timing::Timer depth_timer("depth");
     ++image_cnt_;
     cv::Mat &current_img = img_[image_cnt_ & 1];
     cv::Mat &last_img = img_[!(image_cnt_ & 1)];

     cv_bridge::CvImagePtr cv_ptr;
     cv_ptr = cv_bridge::toCvCopy(depth_queue_.front(), depth_queue_.front()->encoding);
     // TODO: make it a parameter
     constexpr double k_depth_scaling_factor = 1000.0;
     if (depth_queue_.front()->encoding==sensor_msgs::image_encodings::TYPE_32FC1) {
          (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor);
     }
     cv_ptr->image.copyTo(current_img);

     double depth;
     cloud_.clear();
     double uu, vv;

     uint16_t *row_ptr;
     int cols = current_img.cols, rows = current_img.rows;
     if (!parameters_.use_depth_filter_) {
          for (int v = 0; v < rows; v++) {
               row_ptr = current_img.ptr<uint16_t>(v);
               for (int u = 0; u < cols; u++) {
                    depth = (*row_ptr++)/k_depth_scaling_factor;
                    pcl::PointXYZ point;
                    point.x = (u - parameters_.center_x_)*depth/parameters_.focal_x_;
                    point.y = (v - parameters_.center_y_)*depth/parameters_.focal_y_;
                    point.z = depth;
                    cloud_.push_back(point);
               }
          }
     } else {
          if (image_cnt_!=1) {
               Eigen::Vector4d coord_h;
               Eigen::Vector3d coord;
               for (int v = parameters_.depth_filter_margin_; v < rows - parameters_.depth_filter_margin_; v++) {
                    row_ptr = current_img.ptr<uint16_t>(v) + parameters_.depth_filter_margin_;
                    for (int u = parameters_.depth_filter_margin_; u < cols - parameters_.depth_filter_margin_; u++) {
                         depth = (*row_ptr++)/k_depth_scaling_factor;
                         pcl::PointXYZ point;
                         point.x = (u - parameters_.center_x_)*depth/parameters_.focal_x_;
                         point.y = (v - parameters_.center_y_)*depth/parameters_.focal_y_;
                         point.z = depth;
                         if (depth > parameters_.depth_filter_max_dist_ || depth < parameters_.depth_filter_min_dist_)
                              continue;
                         coord_h = last_transform_.inverse()*transform_*Eigen::Vector4d(point.x, point.y, point.z, 1);
                         coord = Eigen::Vector3d(coord_h(0), coord_h(1), coord_h(2))/coord_h(3);
                         uu = coord.x()*parameters_.focal_x_/coord.z() + parameters_.center_x_;
                         vv = coord.y()*parameters_.focal_y_/coord.z() + parameters_.center_y_;
                         if (uu >= 0 && uu < cols && vv >= 0 && vv < rows) {
//                        getInterpolation(last_img, uu, vv)
                              if (fabs(last_img.at<uint16_t>((int) vv, (int) uu)/k_depth_scaling_factor - coord.z())
                                  < parameters_.depth_filter_tolerance_) {
                                   cloud_.push_back(point);
                              }
                         } //else cloud_.push_back(point_);
                    }
               }
          }
     }
     depth_timer.Stop();
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::SynchronizationAndProcess() {
     ros::Time depth_time;
     double time_delay = 3e-3;
     while (!depth_queue_.empty()) {
          bool new_pos = false;
          depth_time = depth_queue_.front()->header.stamp;
          while (transform_queue_.size() > 1 &&
              std::get<0>(transform_queue_.front()) <= depth_time + ros::Duration(time_delay)) {
               sync_pos_ = std::get<1>(transform_queue_.front());
               sync_q_ = std::get<2>(transform_queue_.front());
               transform_queue_.pop();
               new_pos = true;
          }
          if (transform_queue_.empty()
              || std::get<0>(transform_queue_.front()) <= depth_time + ros::Duration(time_delay)) {
               break;
          }
          if (!new_pos) {
               depth_queue_.pop();
               continue;
          }

          new_msg_ = true;
#ifndef PROBABILISTIC
          // TODO: sync_depth_ must be PointCloud2
            sync_depth_ = depth_queue_.front();
            return;
#else
          if (parameters_.use_depth_filter_)
               last_transform_ = transform_;
          transform_.block<3, 3>(0, 0) = sync_q_.toRotationMatrix();
          transform_.block<3, 1>(0, 3) = sync_pos_;
          transform_(3, 0) = transform_(3, 1) = transform_(3, 2) = 0;
          transform_(3, 3) = 1;
          transform_ = transform_*parameters_.T_D_B_*parameters_.T_B_C_;
          raycast_origin_ = Eigen::Vector3d(transform_(0, 3), transform_(1, 3), transform_(2, 3))/transform_(3, 3);

          if constexpr(std::is_same<DepthMsgType, sensor_msgs::Image::ConstPtr>::value) {
               DepthConversion();
          } else if constexpr(std::is_same<DepthMsgType, sensor_msgs::PointCloud2::ConstPtr>::value) {
               sensor_msgs::PointCloud2::ConstPtr tmp = depth_queue_.front();
               pcl::fromROSMsg(*tmp, cloud_);
          }

          std::cout << "Pointcloud Size:\t" << cloud_.points.size() << std::endl;
          if ((int) cloud_.points.size()==0) {
               depth_queue_.pop();
               continue;
          }

          RaycastMultithread();
          depth_queue_.pop();
#endif
     }
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::PoseCallback(const PoseMsgType &msg) {
     Eigen::Vector3d pos;
     Eigen::Quaterniond q;

     if constexpr(std::is_same<PoseMsgType, geometry_msgs::PoseStamped::ConstPtr>::value) {
          pos = Eigen::Vector3d(msg->pose.position.x,
                                msg->pose.position.y,
                                msg->pose.position.z);
          q = Eigen::Quaterniond(msg->pose.orientation.w,
                                 msg->pose.orientation.x,
                                 msg->pose.orientation.y,
                                 msg->pose.orientation.z);
     } else if constexpr(std::is_same<PoseMsgType, nav_msgs::Odometry::ConstPtr>::value) {
          pos = Eigen::Vector3d(msg->pose.pose.position.x,
                                msg->pose.pose.position.y,
                                msg->pose.pose.position.z);
          q = Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                 msg->pose.pose.orientation.x,
                                 msg->pose.pose.orientation.y,
                                 msg->pose.pose.orientation.z);
     } else if constexpr(std::is_same<PoseMsgType, geometry_msgs::TransformStamped::ConstPtr>::value) {
          pos = Eigen::Vector3d(msg->transform.translation.x,
                                msg->transform.translation.y,
                                msg->transform.translation.z);
          q = Eigen::Quaterniond(msg->transform.rotation.w,
                                 msg->transform.rotation.x,
                                 msg->transform.rotation.y,
                                 msg->transform.rotation.z);
     }

     transform_queue_.push(std::make_tuple(msg->header.stamp, pos, q));
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::PathCallback(const nav_msgs::Path::ConstPtr &path){
     for(int i=0;i<path->poses.size();i++){
               path_pos_=Eigen::Vector3d(path->poses[i].pose.position.x,
                                         path->poses[i].pose.position.y,
                                         path->poses[i].pose.position.z);
               path_q_=Eigen::Vector4d(path->poses[i].pose.orientation.w,
                                          path->poses[i].pose.orientation.x,
                                          path->poses[i].pose.orientation.y,
                                          path->poses[i].pose.orientation.z);
     }
}
template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::DepthCallback(const DepthMsgType &depth_map) {
     depth_queue_.push(depth_map);
     SynchronizationAndProcess();
}

template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::UpdateEsdfEvent(const ros::TimerEvent & /*event*/) {
     if (!new_msg_)
          return;
     new_msg_ = false;
     cur_pos_ = sync_pos_;

#ifndef PROBABILISTIC
     timing::Timer handlePCTimer("handlePointCloud");
       pcl::fromROSMsg(*sync_pc_, cloud_);

       esdf_map_->SetUpdateRange(cur_pos_ - parameters_.radius_, cur_pos_ + parameters_.radius_, false);
       esdf_map_->SetAway();

       Eigen::Vector3i tmp_vox;
       Eigen::Vector3d tmp_pos;
       for (int i = 0; i < cloud_.size(); i++) {
         tmp_pos = Eigen::Vector3d(cloud_[i].x, cloud_[i].y, cloud_[i].z);
         esdf_map_->SetOccupancy(tmp_pos, 1);
       }
       esdf_map_->SetBack();
       handlePCTimer.Stop();
#endif
     esdf_cnt_++;
     std::cout << "Running " << esdf_cnt_ << " updates." << std::endl;
//    ros::Time t1 = ros::Time::now();
     if (esdf_map_->CheckUpdate()) {
          timing::Timer update_esdf_timer("UpdateESDF");
          if (parameters_.global_update_/*true*/)
               esdf_map_->SetOriginalRange();
          else
               esdf_map_->SetUpdateRange(cur_pos_ - parameters_.radius_, cur_pos_ + parameters_.radius_);
          esdf_map_->UpdateOccupancy(parameters_.global_update_);
          esdf_map_->UpdateESDF();
#ifdef SIGNED_NEEDED
          // TODO: Complete this SIGNED_NEEDED
            inv_esdf_map_->UpdateOccupancy();
            inv_esdf_map_->UpdateESDF();
#endif
          update_esdf_timer.Stop();
          timing::Timing::Print(std::cout);
     }
//    ros::Time t2 = ros::Time::now();

//    std::string text = "Fiesta\nCurrent update Time\n"
//                       + timing::Timing::SecondsToTimeString((t2 - t1).toSec() * 1000)
//                       + " ms\n" + "Average update Time\n" +
//                       timing::Timing::SecondsToTimeString(timing::Timing::GetMeanSeconds("UpdateESDF") * 1000)
//                       + " ms";

     if (parameters_.visualize_every_n_updates_!=0 && esdf_cnt_%parameters_.visualize_every_n_updates_==0) {
//        std::thread(Visualization, esdf_map_, text).detach();
          Visualization(esdf_map_, parameters_.global_vis_, "");
     }
//    else {
//        std::thread(Visualization, nullptr, text).detach();
//        Visualization(nullptr, globalVis, "");
//    }
}
template<class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::publishRobotMesh() {
  // publish the robot model with the pose
  visualization_msgs::Marker robot_model;
  robot_model.header.frame_id = "world";
  robot_model.header.stamp = ros::Time();
  robot_model.mesh_resource = "file://" + parameters_.robot_model_file_;
  robot_model.mesh_use_embedded_materials = true;
  robot_model.scale.x = robot_model.scale.y = robot_model.scale.z =
      parameters_.resolution_*6;
  robot_model.lifetime = ros::Duration();
  robot_model.action = visualization_msgs::Marker::MODIFY;
  robot_model.color.a = robot_model.color.r = robot_model.color.g =
      robot_model.color.b = 1.;
  robot_model.type = visualization_msgs::Marker::MESH_RESOURCE;

  // Change to horizontal camera frame
//   Transformation T_G_CH = T_G_C * transformer_.getModelTransform();
//   Eigen::Quaternionf quatrot = T_G_CH.getEigenQuaternion();
//   Point quat_vec = quatrot.vec();
//   robot_model.pose.orientation.x = sync_pos_(0);
//   robot_model.pose.orientation.y = sync_pos_(1);
//   robot_model.pose.orientation.z = sync_pos_(2);
//   robot_model.pose.orientation.w = sync_pos_(3);
  robot_model.pose.orientation.x = path_q_(1);
  robot_model.pose.orientation.y = path_q_(2);
  robot_model.pose.orientation.z = path_q_(3);
  robot_model.pose.orientation.w = path_q_(0);
//   Point translation = T_G_CH.getPosition();
  robot_model.pose.position.x = path_pos_(0)-parameters_.little_x_;
  robot_model.pose.position.y = path_pos_(1);
  robot_model.pose.position.z = path_pos_(2)+parameters_.little_z_;
//   cur_pos_写入到txt文档中
     std::ofstream write;
     write.open("/home/byl/Desktop/fast_Planner_VDB/fiesta_ws/src/Fiesta/pose.txt",std::ios::app);
     if(!write.is_open())
     {
          std::cerr<<"\033[31m"<<"Error: failed to open pose.txt"<<"\033[0m"<<std::endl;
     }else{
          // std::cerr<<"\033[33m 文件打开成功！\033[0m"<<std::endl;  // 经测试，文件是可以打开的
         write << cur_pos_(0) << " " << cur_pos_(1) << " " << cur_pos_(2) << std::endl;
     }
     write.close();
  robot_model_pub_.publish(robot_model);
}
}
#endif //ESDF_TOOLS_INCLUDE_FIESTA_H_
