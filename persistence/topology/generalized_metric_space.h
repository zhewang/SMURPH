#ifndef GENERALIZEDMETRICSPACE_H
#define GENERALIZEDMETRICSPACE_H

#include "metric_space.h"
#include <vector>

class GeneralizedMetricSpace : public MetricSpace {
	public:
		GeneralizedMetricSpace(int _numPoints, double** _distances);
		~GeneralizedMetricSpace();

		virtual double distance(int _i, int _j);

	private:
		double** distances;
};

#endif
