#include "euclidean_space.h"

EuclideanSpace::EuclideanSpace(const Points & _points) : MetricSpace(_points.size())  {
	points = _points;
}

EuclideanSpace::~EuclideanSpace()  {
}

double EuclideanSpace::distance(int _i, int _j)  {
	return points[_i].length(points[_j]);
}
