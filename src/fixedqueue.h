#ifndef FIXEDQUEUE_H
#define FIXEDQUEUE_H

#include <deque>

template <typename T>
class FixedQueue
{
public:

    typedef typename std::deque<T>::iterator iterator;
    typedef typename std::deque<T>::const_iterator const_iterator;

    //set max size
    FixedQueue(size_t size)
    {
        que.resize(size);
    }

    //push back
    void push_back(const T&& t)
    {
        que.pop_front();
        que.push_back(t);
    }

    void push_back(const T& t)
    {
        que.pop_front();
        que.push_back(t);
    }

    //push front
    void push_front(const T&& t)
    {
        que.pop_back();
        que.push_front(t);
    }

    void push_front(const T& t)
    {
        que.pop_back();
        que.push_front(t);
    }


    //access front
    T & get_front(void)
    {
        return que.front();
    }

    //access back
    T & get_back(void)
    {
        return que.back();
    }
    //Get size
    int get_size(void)
    {
        return que.size();
    }

    iterator begin() { return que.begin(); }
    const_iterator begin() const { return que.begin(); }

    iterator end() { return que.end(); }
    const_iterator end() const { return que.end(); }

private:

    std::deque < T > que;       //data container
};



#endif // FIXEDQUEUE_H
