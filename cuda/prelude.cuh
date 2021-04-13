
template<typename T>
std::unique_ptr<T, gwh_deleter_host<T>> gwh_allocate_host(size_t len) {
    return nullptr;
}

template<typename T>
std::unique_ptr<T, gwh_deleter_host<T>> gwh_allocate_device(size_t len) {
    return nullptr;
}

template<typename T>
void gwh_copy_host(T* dst, const T* src, size_t len) {

}

template<typename T>
void gwh_copy_device(T* dst, const T* src, size_t len) {
    
}

template<typename T>
T gwh_read_at(const T* array, size_t index) {
    return T();
}

template<typename T>
void gwh_write_at(T* array, size_t index, T val) {

}

template<typename T>
struct gwh_deleter_host {
    void operator()(const T* array) {

    }
}

template<typename T>
struct gwh_deleter_device {
    void operator()(const T* array) {
        
    }
}