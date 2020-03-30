#include "heap.h"


BinaryHeap::BinaryHeap(const vector<int> &vector) : __heap(vector)
{
    Heapify();
}


void BinaryHeap::Heapify()
{
    size_t length = __heap.size();
    for (size_t i = length - 1; i >= 0; --i)
    {
        SiftDown(i);
    }
}


void BinaryHeap::SiftDown(size_t index)
{
    size_t length = __heap.size();
    size_t leftIndex = 2 * index + 1;
    size_t rightIndex = 2 * index + 2;

    if (leftIndex >= length)
    {
        return;
    }

    size_t minIndex = index;
    if (__heap[index] > __heap[leftIndex])
    {
        minIndex = leftIndex;
    }
    if ((rightIndex < length) && (__heap[minIndex] > __heap[rightIndex]))
    {
        minIndex = rightIndex;
    }
    if (minIndex != index)
    {
        swap(__heap[index], __heap[minIndex]);
        SiftDown(minIndex);
    }
}


void BinaryHeap::SiftUp(size_t index)
{
    if (index == 0)
    {
        return;
    }

    size_t parentIndex = (index - 1) / 2;
    if (__heap[parentIndex] > __heap[index])
    {
        swap(__heap[parentIndex], __heap[index]);
        SiftUp(parentIndex);
    }
}


void BinaryHeap::Insert(int value)
{
    size_t length = __heap.size();
    __heap.push_back(value);
    SiftUp(length);
}


int BinaryHeap::GetMin()
{
    return __heap[0];
}


void BinaryHeap::DeleteMin()
{
    size_t length = __heap.size();

    if (length == 0)
    {
        return;
    }

    __heap[0] = __heap[length - 1];
    __heap.pop_back();

    SiftDown(0);
}


size_t BinaryHeap::Size()
{
    return __heap.size();
}


void BinaryHeap::LazyInsert(int element)
{
    __heap.push_back(element);
}


void BinaryHeap::DisplayHeap()
{
    cout << "Heap: ";
    for (auto &it : __heap)
    {
        cout << it << "; ";
    }
    cout << endl;
}


ostream &operator<<(ostream &os, const BinaryHeap &heap)
{
    os << "Heap: ";
    for (auto &it : heap.__heap)
    {
        os << it << "; ";
    }
    os << '\n';
    return os;
}