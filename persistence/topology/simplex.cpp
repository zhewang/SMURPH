#include "simplex.h"

Simplex::Simplex(const std::vector<int> & _simplex, MetricSpace* _metricSpace)  {
	simplex = _simplex;
	metric_space = _metricSpace;
	cached_distance = 0;
}

Simplex::~Simplex()  {
}

std::vector<Simplex> Simplex::faces()  {
	std::vector<Simplex> all_faces;
	for(unsigned i = 0; i < simplex.size(); i++)  {
		std::vector<int> new_face;
		for(unsigned j = 0; j < simplex.size()-1; j++)  {
			int next_vertex = simplex[(j+i)%simplex.size()];
			new_face.push_back(next_vertex);
		}
		all_faces.push_back(Simplex(new_face,metric_space));
	}
	return all_faces;
}
