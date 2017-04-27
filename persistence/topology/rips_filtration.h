#ifndef RIPSFILTRATION_H
#define RIPSFILTRATION_H

#include "filtration.h"
#include "generalized_metric_space.h"

class RipsFiltration : public Filtration  {
	public:
		RipsFiltration(const Points & _points, int _maxD);
		RipsFiltration(int _numPoints, double** _distanceMat, int _maxD);
		RipsFiltration(int _numPoints, int _maxD);
		~RipsFiltration();

		void global_bron_kerbosch(std::vector<Simplex>* _rips, std::vector<int> _R, const std::set<int> & _P, int _maxD);

		virtual bool build_filtration();


    void set_distance(int _i, int _j, double _v);
    void build_metric();

	protected:
		int num_points;
		double** distances;
		MetricSpace* metric_space;

	private:
};

#endif
