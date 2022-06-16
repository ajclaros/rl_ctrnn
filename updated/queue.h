#include <iostream>
#include <cstdlib>

// Define the default capacity of the queue
// A class to represent a queue
template<class T>
class queue
{
     T *arr;         // array to store queue elements
    int front;      // front points to the front element in the queue (if any)
    int rear;       // rear points to the last element in the queue
    int capacity;

public:
    queue(int s);
    queue(int, T initial);

    T pop();
    void push(T x);
    T peek();
    int size();
    bool isEmpty();
    bool isFull();
    double avg();
    void print();
};


template <class T>
queue<T>::queue(int size)
{
    arr = new T[size];
    capacity = size;
    front = -1;
    rear = -1;
}
template <class T>
queue<T>::queue(int s, T initial)
{
    arr = new T[s];
    capacity = s;
    for(int i=0; i<capacity; i++)
    {arr[i] = initial;}
    front = -1;
    rear = -1;
}
// Utility function to dequeue the front element
template <class T> T queue<T>::pop()
{
    // check for queue underflow
    if (isEmpty())
    {
        std::cout << "Underflow\nProgram Terminated\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Removing " << arr[front] << std::endl;
    arr[front] = 0;

    front = (front + 1) % capacity;
    return (arr[front-1%capacity]);

}

// Utility function to add an item to the queue
template <class T>
void queue<T>::push(T item)
{
    // check for queue overflow

    arr[rear] = item;
    rear = (rear + 1) % capacity;
}

// Utility function to return the front element of the queue
template <class T> T queue<T>::peek()
{
    if (isEmpty())
    {
        std::cout << "UnderFlow\nProgram Terminated\n";
        exit(EXIT_FAILURE);
    }
    return arr[front];
}

// Utility function to return the size of the queue
template <class T> int queue<T>::size() {
    return capacity;
}

// Utility function to check if the queue is empty or not
template <class T> bool queue<T>::isEmpty() {
    return (capacity==0);
}
template <class T> void queue<T>::print()
{
    for (int i=0; i<capacity; i++)
    {
        std::cout<<arr[(front+i)%capacity] <<" ";
    }
    std::cout<<std::endl;

}
template <class T> double queue<T>::avg()
{
    double sum=0;
    for(int i=0; i<capacity; i++){
        sum+=arr[i];
}
    return sum/capacity;

}
