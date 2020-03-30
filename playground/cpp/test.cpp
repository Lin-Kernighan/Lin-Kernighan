#include <vector>
#include <iostream>

#include "heap.h"

using std::vector;
using std::cout;
using std::endl;

int main() {
    vector<int> test_vector{10, 8, 6, 4, 2, 1};
    BinaryHeap test_heap(test_vector);
    test_heap.DisplayHeap();
}