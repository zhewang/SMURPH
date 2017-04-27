#ifndef METRICSPACE_H
#define METRICSPACE_H

class MetricSpace  {
	public:
		MetricSpace(int _numPoints);
		~MetricSpace();

		virtual double distance(int _i, int _j) = 0;

	private:
		int num_points;
};

#endif
