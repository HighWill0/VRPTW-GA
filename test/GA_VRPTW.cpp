#include <bits/stdc++.h>
using namespace std;

const int NUM_CUSTOMERS = 100;
const int POP_SIZE = 100;
const int MAX_GEN = 1000;
const double MUTATION_RATE = 0.4;
const double CROSSOVER_RATE = 0.75;

struct Customer {
    int id;
    double x, y;
    double demand, readyTime, dueTime, serviceTime;
};

vector<Customer> customers;
vector<vector<double>> distanceMatrix;

struct Route {
    vector<int> customers;
    double cost = 0;
};

struct Individual {
    vector<Route> routes;
    double totalCost = 0;
};

double euclidean(const Customer& a, const Customer& b) {
    return hypot(a.x - b.x, a.y - b.y);
}

void computeDistanceMatrix() {
    distanceMatrix.resize(customers.size(), vector<double>(customers.size()));
    for (int i = 0; i < customers.size(); i++)
        for (int j = 0; j < customers.size(); j++)
            distanceMatrix[i][j] = euclidean(customers[i], customers[j]);
}

Individual createGreedyIndividual() {
    Individual ind;
    vector<bool> visited(customers.size(), false);
    visited[0] = true;

    while (count(visited.begin(), visited.end(), false) > 0) {
        Route route;
        int current = 0;
        double time = 0;
        while (true) {
            int next = -1;
            double best = 1e9;
            for (int i = 1; i < customers.size(); i++) {
                if (!visited[i]) {
                    double arrival = time + distanceMatrix[current][i];
                    if (arrival >= customers[i].readyTime &&
                        arrival <= customers[i].dueTime &&
                        distanceMatrix[current][i] < best) {
                        next = i;
                        best = distanceMatrix[current][i];
                    }
                }
            }
            if (next == -1) break;
            visited[next] = true;
            route.customers.push_back(next);
            time += best + customers[next].serviceTime;
            route.cost += best;
            current = next;
        }
        route.cost += distanceMatrix[current][0]; // về depot
        ind.routes.push_back(route);
    }

    ind.totalCost = 0;
    for (auto& r : ind.routes)
        ind.totalCost += r.cost;
    return ind;
}

void mutate(Individual& ind) {
    if (ind.routes.size() < 2) return;
    int r1 = rand() % ind.routes.size();
    int r2 = rand() % ind.routes.size();
    if (ind.routes[r1].customers.empty() || ind.routes[r2].customers.empty()) return;

    int i1 = rand() % ind.routes[r1].customers.size();
    int i2 = rand() % ind.routes[r2].customers.size();
    swap(ind.routes[r1].customers[i1], ind.routes[r2].customers[i2]);
}

Individual crossover(const Individual& a, const Individual& b) {
    Individual child = a;
    if (a.routes.size() != b.routes.size()) return child;
    for (size_t i = 0; i < a.routes.size(); i++) {
        if ((rand() % 100) / 100.0 < 0.5)
            child.routes[i] = b.routes[i];
    }
    child.totalCost = 0;
    for (auto& r : child.routes) {
        r.cost = 0;
        int prev = 0;
        for (int c : r.customers) {
            r.cost += distanceMatrix[prev][c];
            prev = c;
        }
        r.cost += distanceMatrix[prev][0];
        child.totalCost += r.cost;
    }
    return child;
}

bool compare(const Individual& a, const Individual& b) {
    return a.totalCost < b.totalCost;
}

void loadSolomonData(const string& file) {
    ifstream in(file);
    string line;
    for (int i = 0; i < 9; ++i) getline(in, line); // skip headers
    customers.clear();
    while (getline(in, line)) {
        if (line.empty()) break;
        stringstream ss(line);
        Customer c;
        ss >> c.id >> c.x >> c.y >> c.demand >> c.readyTime >> c.dueTime >> c.serviceTime;
        customers.push_back(c);
    }
}

int main() {
    srand(time(0));
    loadSolomonData("C101.txt");
    computeDistanceMatrix();

    vector<Individual> population;
    for (int i = 0; i < POP_SIZE; i++)
        population.push_back(createGreedyIndividual());

    for (int gen = 0; gen < MAX_GEN; gen++) {
        sort(population.begin(), population.end(), compare);
        vector<Individual> newPop;
        newPop.push_back(population[0]);

        while (newPop.size() < POP_SIZE) {
            Individual p1 = population[rand() % POP_SIZE];
            Individual p2 = population[rand() % POP_SIZE];
            Individual child = crossover(p1, p2);
            if ((rand() % 100) / 100.0 < MUTATION_RATE)
                mutate(child);
            newPop.push_back(child);
        }
        population = newPop;
    }

    sort(population.begin(), population.end(), compare);
    Individual best = population[0];
    int routeId = 1;
    for (auto& r : best.routes) {
        cout << "Route #" << routeId++ << ": ";
        for (int c : r.customers) cout << c << " ";
        cout << "\n";
    }
    cout << "Cost " << best.totalCost << "\n";
    return 0;
}
