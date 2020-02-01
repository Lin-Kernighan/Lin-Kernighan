#include "heap.h"

template <class T>
BinaryHeap<T>::BinaryHeap(const vector<T> &vector) : __heap(vector)
{
    Heapify();
}

template <class T>
void BinaryHeap<T>::Heapify()
{
    size_t length = __heap.size();
    for (size_t i = length - 1; i >= 0; --i)
    {
        SiftDown(i);
    }
}

template <class T>
void BinaryHeap<T>::SiftDown(size_t index)
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

template <class T>
void BinaryHeap<T>::SiftUp(size_t index)
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

template <class T>
void BinaryHeap<T>::Insert(T value)
{
    size_t length = __heap.size();
    __heap.push_back(value);
    SiftUp(length);
}

template <class T>
T BinaryHeap<T>::GetMin()
{
    return __heap[0];
}

template <class T>
void BinaryHeap<T>::DeleteMin()
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

template <class T>
size_t BinaryHeap<T>::Size()
{
    return __heap.size();
}

template <class T>
void BinaryHeap<T>::LazyInsert(T element)
{
    __heap.push_back(element);
}

template <class T>
void BinaryHeap<T>::DisplayHeap()
{
    cout << "Heap: ";
    for (auto &it : __heap)
    {
        cout << it << "; ";
    }
    cout << endl;
}

template <class T>
ostream &operator<<(ostream &os, const BinaryHeap<T> &heap)
{
    os << "Heap: ";
    for (auto &it : heap.__heap)
    {
        os << *it << "; ";
    }
    os << '\n';
}