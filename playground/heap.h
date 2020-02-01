#ifndef STRUCTURES_BINARYHEAP_H
#define STRUCTURES_BINARYHEAP_H

#include <vector>
#include <iostream>

using std::cout;
using std::endl;
using std::ostream;
using std::swap;
using std::vector;

class BinaryHeap
{
private:
    vector<int> __heap;

    void SiftDown(size_t index);
    void SiftUp(size_t index);
    void Heapify();

public:
    explicit BinaryHeap() = default;
    explicit BinaryHeap(const BinaryHeap &heap) = default;
    explicit BinaryHeap(const vector<int> &data);
    ~BinaryHeap() = default;

    friend ostream &operator<<(ostream &stream, const BinaryHeap &heap);

    void Insert(int element);
    void LazyInsert(int element);
    void DeleteMin();
    void DisplayHeap();
    size_t Size();
    int GetMin();
};

#endif //STRUCTURES_BINARYHEAP_H
