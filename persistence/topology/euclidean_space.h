#ifndef EUCLIDEANSPACE_H
#define EUCLIDEANSPACE_H

#include "metric_space.h"
#include "../geometry/point_incs.h"

class EuclideanSpace : public MetricSpace {
	public:
		EuclideanSpace(const Points & _points);
		~EuclideanSpace();

		virtual double distance(int _i, int _j);

	private:
		Points points;
};

#endif
