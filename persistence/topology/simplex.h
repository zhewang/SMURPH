#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

#include "metric_space.h"

class Simplex  {
	public:
		Simplex()  {}
		Simplex(const std::vector<int> & _simplex, MetricSpace* _metricSpace);
		~Simplex();

		int dim() const { return simplex.size()-1; }
		int vertex(int _i) const { return simplex[_i]; }
		double get_simplex_distance() const  {
			return cached_distance;
		}

		void compute_simplex_distance()  {
			for(unsigned i = 0; i < simplex.size(); i++)  {
				for(unsigned j = 0; j < i; j++)  {
					double next_dist = metric_space->distance(simplex[i],simplex[j]);
					cached_distance = next_dist > cached_distance ? next_dist : cached_distance;
				}
			}
		}

		std::string unique_unoriented_id() const  {
			std::vector<int> sorted_simplex = simplex;
			std::sort(sorted_simplex.begin(),sorted_simplex.end());
			char unique_id[10*(sorted_simplex.size()+1)];
			sprintf(unique_id, "%u", sorted_simplex[0]);
			for(unsigned i = 1; i < sorted_simplex.size(); i++)
				sprintf(unique_id, "%s-%u", unique_id, sorted_simplex[i]);
			return unique_id;
		}

		std::vector<Simplex> faces();

		inline bool operator<(const Simplex& _simp) const {
			int our_dim = this->dim(), other_dim = _simp.dim();
			// lower dimensional simplices come before higher dimensional ones
			/*
			if(our_dim < other_dim)
				return true;
			else if(other_dim < our_dim)
				return false;
				*/

			// if dimensions are 0, then just use index for comparison
			if(our_dim==0 && other_dim==0)
				return simplex[0] < _simp.vertex(0);

			// otherwise, take largest edge distance on simplex for comparison
			double our_edge_dist = this->get_simplex_distance(), other_edge_dist = _simp.get_simplex_distance();
			if(our_edge_dist < other_edge_dist)
				return true;
			if(other_edge_dist < our_edge_dist)
				return false;

			// if they are equivalent, then lower dimensional simplices precede higher dimensional ones
			return our_dim < other_dim;

			// TODO: resolve equal distance, equal dim?
		}

		friend std::ostream& operator <<(std::ostream &out, const Simplex & _simplex)  {
			for(int i = 0; i <= _simplex.dim(); i++)  {
				out << " " << _simplex.vertex(i);
			}
			return out;
		}

	private:
		std::vector<int> simplex;
		MetricSpace* metric_space;
		double cached_distance;
};

#endif
