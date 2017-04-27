#ifndef RELAXEDMETRICSPACE_H
#define RELAXEDMETRICSPACE_H

#include "metric_space.h"
#include <vector>
#include <map>
#include "../geometry/point_incs.h"

class RelaxedMetricSpace : public MetricSpace {
	public:
		RelaxedMetricSpace(const SparsePoints & _points, const std::vector<double> & _deletionTimes, double _eps);
		~RelaxedMetricSpace();

		double get_time_scale()  { return time_scale; }
		void set_time_scale(double _scale)  { time_scale = _scale; }

		double get_deletion_time(int _i)  { return time_scale*deletion_times[_i]; }

		virtual double distance(int _i, int _j);

	private:
		SparsePoints points;
		std::vector<double> deletion_times;
		std::map<std::pair<int,int>, double> distance_map;
		double** cached_distances;
		double eps;
		double time_scale;
};

#endif
