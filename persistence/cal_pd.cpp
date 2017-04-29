#include <iostream>
#include <cstdlib>
#include <string>

#include "json.hpp"

#include "geometry/covertree.h"
#include "topology/sparse_rips_filtration.h"
#include "topology/rips_filtration.h"
#include "topology/persistence.h"


using json = nlohmann::json;

Points read_points_from_json(std::string str, int dim)
{
    Points points;
    json data = json::parse(str);
    //for(auto it = data.begin(); it != data.end(); it ++) {
    for(auto &t : data) {
        vector<double> p;
        p.push_back(t[0]);
        p.push_back(t[1]);
        if(dim == 3) {
            p.push_back(t[2]);
        }
        points.push_back(Vector(p));
    }
    return points;
}

int main(int argc, char** argv)  {

    if(argc != 3) {
        std::cout << "Usage: " << argv[0];
        std::cout << " [point dimension (2 or 3)] [data as json string]\n";
        return 0;
    }
    int point_dim = std::stoi(argv[1]);
    std::string str = argv[2];
    Points points = read_points_from_json(str, point_dim);

	int max_d = 2;
    //Filtration* filtration = new SparseRipsFiltration(points, max_d, 1.0/3);
    Filtration* filtration = new RipsFiltration(points, max_d);
    PersistentHomology ph(filtration);
    PersistenceDiagram *pd = ph.compute_persistence();

    pd->sort_pairs_by_persistence();
    for(unsigned i = 0; i < pd->num_pairs(); i++)  {
        PersistentPair pairing = pd->get_pair(i);
        printf("%u %.7f %.7f\n", pairing.dim(), pairing.birth_time(), pairing.death_time());
    }
}
