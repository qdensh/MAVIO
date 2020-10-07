#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include<math.h>
using namespace std;
using namespace Eigen;

// raw_gnss fusion
inline vector<pair<int, int>> GetMatchingTime(
	const vector<pair<double, Vector3d>>& pts1,
	const vector<pair<double, Vector3d>>& pts2)
{
	vector<pair<int, int>> vec;
	int i = 0, j = 0;
	double dt = 1000000000;
	while (i < pts1.size() && j < pts2.size())
	{
		if (pts1[i].first < pts2[j].first)
		{
			if (pts2[j].first - pts1[i].first < dt + 0.00001)
			{
				dt = pts2[j].first - pts1[i].first;
				i++;
				continue;
			}
			else
			{
				vec.push_back(pair<int, int>(i, j - 1));
				i++;
				dt = 1000000000;
				continue;
			}
		}
		else
		{
			if (pts1[i].first - pts2[j].first < dt + 0.00001)
			{
				dt = pts1[i].first - pts2[j].first;
				j++;
				continue;
			}
			else
			{
				vec.push_back(pair<int, int>(i - 1, j));
				j++;
				dt = 1000000;
				continue;
			}
		}
	}
	return vec;
}


inline bool registration_4DOF(
	const vector<pair<double, Vector3d>>& t1,
	const vector<pair<double, Vector3d>>& t2,
 	Matrix3d& R , Vector3d &t)
{
	vector<pair<int, int >> v = GetMatchingTime(t1, t2);
	Vector3d v1=t1[v.back().first].second - t1[v.front().first].second;
	Vector3d v2=t2[v.back().second].second - t2[v.front().second].second;
	ROS_INFO("v1 v2 %f %f",v1.norm(),v2.norm());
	if(v1.norm()/v2.norm() < 2 && v1.norm()/v2.norm()>0.2 && v1.norm()>3)
	{
		double yaw=atan2(v2(1),v2(0))-atan2(v1(1),v1(0));
		R<<cos(yaw),-sin(yaw),0,
			sin(yaw),cos(yaw),0,
			0,0,1;
		t = t2[v.back().second].second - R*t1[v.back().first].second;
		cout<< "yaw(deg): " << yaw * 180/M_PI <<endl;
		cout << "t(m): " << t <<endl;
		return true;
	}
	else return false;
}
