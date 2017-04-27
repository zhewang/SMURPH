#include "persistence_diagram.h"

PersistenceDiagram::PersistenceDiagram(std::string _filename)  {
	Points points;
	PointsIO::read_points_from_file(_filename, points);
	for(int i = 0; i < points.size(); i++)  {
		Vector pt = points[i];
		int dim = pt[0];
		double birth = pt[1];
		double death = pt[2];
		PersistentPair persistence_pair(dim,birth,death);

		if(pt.dim() > 3)  {
			for(int j = 3; j < pt.dim(); j++)
				persistence_pair.add_generator_pt(pt[j]);
		}
		all_pairs.push_back(persistence_pair);
	}
	_size_satisfied = true;
}

PersistenceDiagram::PersistenceDiagram(const std::vector<PersistentPair> & _allPairs)  {
	all_pairs = _allPairs;
	_size_satisfied = true;
}

PersistenceDiagram::~PersistenceDiagram()  {
	_size_satisfied = true;
}

std::pair<std::vector<int>,std::vector<int> > PersistenceDiagram::wasserstein_matching(const PersistenceDiagram & _otherPD, int _d, double _p) const  {
	// construct matching matrix
	std::vector<PersistentPair> our_pairs = this->get_persistence_pairs(_d);
	std::vector<PersistentPair> other_pairs = _otherPD.get_persistence_pairs(_d);
	int max_size = our_pairs.size() + other_pairs.size();

	Eigen::MatrixXd cost_matrix = Eigen::MatrixXd::Zero(max_size,max_size);
	// connect all pairs of points
	for(int i = 0; i < our_pairs.size(); i++)  {
		PersistentPair our_pair = our_pairs[i];
		for(int j = 0; j < other_pairs.size(); j++)  {
			PersistentPair other_pair = other_pairs[j];
			cost_matrix(i,j) = pow(our_pair.l_inf_norm(other_pair),_p);
		}
	}
	// connect our pairs to diagonal
	for(int i = 0; i < our_pairs.size(); i++)  {
		PersistentPair our_pair = our_pairs[i];
		for(int j = other_pairs.size(); j < max_size; j++)
			cost_matrix(i,j) = pow(our_pair.l_inf_diagonal(),_p);
	}

	// connect other pairs to diagonal
	for(int j = 0; j < other_pairs.size(); j++)  {
		PersistentPair other_pair = other_pairs[j];
		for(int i = our_pairs.size(); i < max_size; i++) {
			cost_matrix(i,j) = pow(other_pair.l_inf_diagonal(),_p);
		}
	}

	// other method ...
	/*
	double** cost_mat = new double*[max_size];
	for(int i = 0; i < max_size; i++)  {
		cost_mat[i] = new double[max_size];
		for(int j = 0; j < max_size; j++)  {
			cost_mat[i][j] = cost_matrix(i,j);
		}
	}

	HungarianDistance hungarian(cost_mat,max_size,false);
	std::cout << "hungarian algorithm..." << std::endl;
	hungarian.compute();
	std::cout << "... done with hungarian algorithm..." << std::endl;

	for(int i = 0; i < max_size; i++)
		delete [] cost_mat[i];
	delete [] cost_mat;
	*/

	// weighted matching via Munkres method
	Munkres munkres;
	munkres.solve(cost_matrix);
	// perform assignment
	std::vector<int> our_assignment(our_pairs.size(),-2);
	std::vector<int> other_assignment(other_pairs.size(),-2);
	for(int i = 0; i < max_size; i++)  {
		for(int j = 0; j < max_size; j++)  {
			// no assignment
			if(cost_matrix(i,j) != 0)
				continue;

			// assignment made between i and j
			if(i < our_pairs.size() && j < other_pairs.size())  {
				our_assignment[i] = j;
				other_assignment[j] = i;
			}

			// assignment of i to diagonal
			else if(i < our_pairs.size())
				our_assignment[i] = -1;

			// assignment of j to diagonal
			else if(j < other_pairs.size())
				other_assignment[j] = -1;
		}
	}

	// verify: did we assign everyone?
	for(int i = 0; i < our_assignment.size(); i++)  {
		if(our_assignment[i] == -2)  {
			std::cout << "our not assigned!!! " << i << std::endl;
			our_assignment[i] = -1;
		}
	}
	for(int i = 0; i < other_assignment.size(); i++)  {
		if(other_assignment[i] == -2)  {
			std::cout << "other not assigned!!! " << i << std::endl;
			other_assignment[i] = -1;
		}
	}

	return std::pair<std::vector<int>, std::vector<int> >(our_assignment,other_assignment);
}

double PersistenceDiagram::wasserstein_distance(const PersistenceDiagram & _otherPD, int _d, double _p) const  {
	std::pair<std::vector<int>,std::vector<int> > matching = this->wasserstein_matching(_otherPD, _d, _p);
	std::vector<int> src_matchings = matching.first, dst_matchings = matching.second;
	double wasserstein_distance = 0;
	// compute src to dist pairings, including to the diagonal
	for(int s = 0; s < src_matchings.size(); s++)  {
		int s_pt = s, d_pt = src_matchings[s];
		PersistentPair s_pairing = this->get_pair(s_pt);
		if(d_pt == -1)  {
			wasserstein_distance += pow(s_pairing.l_inf_diagonal(),_p);
			continue;
		}
		PersistentPair d_pairing = _otherPD.get_pair(d_pt);
		wasserstein_distance += pow(s_pairing.l_inf_norm(d_pairing),_p);
	}
	int num_dst_diagonals = 0;
	for(int d = 0; d < dst_matchings.size(); d++)  {
		int d_pt = d, s_pt = dst_matchings[d];
		PersistentPair d_pairing = _otherPD.get_pair(d_pt);
		if(s_pt == -1)  {
			wasserstein_distance += pow(d_pairing.l_inf_diagonal(),_p);
			num_dst_diagonals++;
		}
	}
	return pow(wasserstein_distance,1/_p) / (src_matchings.size()+num_dst_diagonals);
	//return pow(wasserstein_distance,1/_p);
}

void PersistenceDiagram::write_to_file(std::string _filename, bool _writeGenerators)  {
	this->sort_pairs_by_persistence();
	FILE* file = fopen(_filename.c_str(), "w");
	for(int i = 0; i < all_pairs.size(); i++)  {
		PersistentPair pair = all_pairs[i];
		if(!_writeGenerators)  {
			fprintf(file, "%u %.7f %.7f\n", pair.dim(), pair.birth_time(), pair.death_time());
			continue;
		}
		fprintf(file, "%u %.7f %.7f", pair.dim(), pair.birth_time(), pair.death_time());
		for(int j = 0; j < pair.generator.size(); j++)
			fprintf(file, " %u", pair.generator[j]);
		fprintf(file, "\n");
	}
	fclose(file);
}
bool PersistenceDiagram::get_size_satisfied() { return this->_size_satisfied; }
