#ifndef FILTRATION_H
#define FILTRATION_H

#include "../geometry/point_incs.h"
#include "simplex.h"
#include <set>

class Filtration  {
	public:
		Filtration(int _maxD);
		~Filtration();

		Simplex get_simplex(int _t)  { return all_simplices[_t]; }
		int filtration_size()  { return all_simplices.size(); }

		int maxD()  { return max_d; }

		virtual bool build_filtration();

	protected:
		std::vector<Simplex> all_simplices;

	private:
		int max_d;
};

#endif
