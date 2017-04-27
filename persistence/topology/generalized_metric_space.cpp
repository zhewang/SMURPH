#include "generalized_metric_space.h"

GeneralizedMetricSpace::GeneralizedMetricSpace(int _numPoints, double** _distances) : MetricSpace(_numPoints) {
	distances = _distances;
}

GeneralizedMetricSpace::~GeneralizedMetricSpace()  {
}

double GeneralizedMetricSpace::distance(int _i, int _j)  {
	return distances[_i][_j];
}
