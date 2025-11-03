#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

class Serializable {
private:
    uint32_t *data;
    int N;

public:
    Serializable(){};
    // Constructor to initialize the data members
    Serializable(uint32_t *data, int N) : data(data), N(N) {}

    // Getter methods for the class
    uint32_t * getData() const { return data; }
    int getLength() const      { return N;  }

    //  Function for Serialization
    void serialize(const std::string& filename)
    {
        std::ofstream file(filename, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open file for writing." << std::endl;
            return;
        }
        
        
        file.write(reinterpret_cast<const char*>(this),
                   sizeof(*this));
        file.close();
        std::cout << "Object serialized successfully." << std::endl;
    }

    //  Function for Deserialization
    static Serializable deserialize(const std::string& filename)
    {
        std::vector<uint32_t> data;
        Serializable obj(data);
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open file for reading." << std::endl;
            return obj;
        }
        file.read(reinterpret_cast<char*>(&obj),
                  sizeof(obj));
        file.close();
        std::cout << "Object deserialized successfully." << std::endl;
        return obj;
    }
};


int main(int argv, char *argc[]) {

    if (argv!=2) {
        std::cerr << "Usage: " << argc[0] << " <input_size>" << std::endl;
        return 1;
    }

    uint32_t N = std::stoi(argc[1]);
    uint32_t elem;
    uint32_t *data = (uint32_t *)calloc(N, sizeof(uint32_t));

    std::ofstream infile;
    infile.open("data.in", std::ios::out | std::ios::trunc);
    infile << "[";
    for (int i = 0; i < N-1; i++) {
        elem = (uint32_t)rand();
        data[i] = elem;
        infile << elem << ", ";
    }
    elem = (uint32_t)rand();
    data[N-1] = elem;
    infile << elem << "]";
    infile.close();

    std::vector<uint32_t> data_vec (data, data+N);

    Serializable original(data_vec);
    original.serialize("data.bin");

    // Deserialize the object
    Serializable restored
        = Serializable::deserialize("data.bin");

    // Test the  deserialized object
    std::vector<uint32_t> data_d = restored.getData();
    std::cout << "Deserialized vector:" << std::endl;
    for (std::vector<uint32_t>::iterator it=data_d.begin(); it!=data_d.end(); ++it) {
        std::cout << ' ' << *it;
    }
    std::cout << std::endl;


    // std::sort (data_vec.begin(), data_vec.end());

    // std::ofstream outfile;
    // outfile.open("sorted.out", std::ios::out | std::ios::trunc);

    // std::cout << "Sorted vector:" << std::endl;
    // for (std::vector<uint32_t>::iterator it=data_vec.begin(); it!=data_vec.end(); ++it) {
    //     std::cout << ' ' << *it;
    // }
    // std::cout << std::endl;

    return 0;
}