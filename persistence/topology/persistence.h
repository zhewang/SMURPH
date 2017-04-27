#ifndef PERSISTENCE_H
#define PERSISTENCE_H

#include "persistence_diagram.h"

#include "filtration.h"
#include "sparse_rips_filtration.h"
#include "rips_filtration.h"
#include <list>
#include <map>
#include "ComputationTimer.h"

typedef std::list<int> PHCycle;

class PersistentHomology  {
	public:
		PersistentHomology(Filtration* _filtration, bool _retainGenerators=false);
		~PersistentHomology();

		PersistenceDiagram *compute_persistence();

		PHCycle merge_cycles(const PHCycle & _c1, const PHCycle & _c2);

	private:
		Filtration* filtration;
		int max_d;
		bool retain_generators;

		std::list<int> expand_chain(int _chain, PHCycle* _allChains);
};

#endif
