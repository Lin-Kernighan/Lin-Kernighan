#ifndef STRUCTURES_BINARYHEAP_H
#define STRUCTURES_BINARYHEAP_H

#include <vector>
#include <iostream>

using std::cout;
using std::endl;
using std::ostream;
using std::swap;
using std::vector;

template <class T>
class BinaryHeap
{
private:
    vector<T> __heap;

    void SiftDown(size_t index);
    void SiftUp(size_t index);
    void Heapify();

public:
    explicit BinaryHeap() = default;
    explicit BinaryHeap(const BinaryHeap &heap) = default;
    explicit BinaryHeap(const vector<T> &data);
    ~BinaryHeap() = default;

    friend ostream &operator<<(ostream &stream, const BinaryHeap<T> &heap);

    void Insert(T element);
    void LazyInsert(T element);
    void DeleteMin();
    void DisplayHeap();
    size_t Size();
    T GetMin();
};

#endif //STRUCTURES_BINARYHEAP_H
