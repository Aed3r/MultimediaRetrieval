#include <igl/opengl/glfw/Viewer.h>
//#include <igl\readPLY.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

#define LABELEDPSB_LOCATION "../data/LabeledDB_new"
#define PRINCETON_LOCATION "../data/psb_v1/benchmark"

enum Facetypes { triangles, quads, mixed };


/*
 * Load a mesh from a OFF file using following specification: https://en.wikipedia.org/wiki/OFF_(file_format)
 */
void load_off(const std::string &filename, Eigen::Ref<Eigen::MatrixXd> V, Eigen::Ref<Eigen::MatrixXi> F, Facetypes type)
{
    std::ifstream file(filename);
    std::string line;
    bool read_counts = false, read_vertices = false;
    int lineCount = 0, num_vertices, num_faces, num_edges, vertexCount = 0, faceCount = 0;
    
    // Get lines until none left
    while (std::getline(file, line))
    {
        // Skip comments, empty lines and the optional 'OFF' header
        if (line[0] == '#' || line.empty() || (lineCount == 0 && line == "OFF"))
            continue;
        
        // Separate line into tokens
        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};

        if (tokens.size() == 3)
        {
            if (!read_counts) {
                // Read counts
                num_vertices = std::stoi(tokens[0]);
                num_faces = std::stoi(tokens[1]);
                num_edges = std::stoi(tokens[2]);

                V.resize(num_vertices, 3);
                F.resize(num_faces, 3);
                read_counts = true;
                continue;
            }

            if (read_vertices)
                throw std::runtime_error("Unexpected vertex line in faces section");

            if (vertexCount >= num_vertices)
                throw std::runtime_error("More vertices in file than expected!");
            
            // Read vertex coordinates
            V(vertexCount, 0) = std::stod(tokens[0]);
            V(vertexCount, 1) = std::stod(tokens[1]);
            V(vertexCount, 2) = std::stod(tokens[2]);
            vertexCount++;
            continue;
        }

        if (tokens.size() == 4)
        {
            if (!read_vertices)
                read_vertices = true;

            if (faceCount >= num_faces)
                throw std::runtime_error("More faces in file than expected!");

            if (!read_counts)
                throw std::runtime_error("File line counts missing!");
            
            if (std::stoi(tokens[0]) == 3) {
                // Read triangle
                F(faceCount, 0) = std::stoi(tokens[1]);
                F(faceCount, 1) = std::stoi(tokens[2]);
                F(faceCount, 2) = std::stoi(tokens[3]);
            } else if (std::stoi(tokens[0]) == 4) {
                // Read quads
                F(faceCount, 0) = std::stoi(tokens[1]);
                F(faceCount, 1) = std::stoi(tokens[2]);
                F(faceCount, 2) = std::stoi(tokens[3]);
                F(faceCount, 3) = std::stoi(tokens[4]);
            } else {
                throw std::runtime_error("Unsupported face type!");
            }
            
            faceCount++;
            continue;
        }

        throw std::runtime_error("Unexpected line in file!");
    }
}





/*
 * Load a mesh from a PLY file using following specification: https://en.wikipedia.org/wiki/PLY_(file_format)#:~:text=PLY%20is%20a%20computer%20file,list%20of%20nominally%20flat%20polygons.
 */
void load_PLY(const std::string& filename, Eigen::Ref<Eigen::MatrixXd> V, Eigen::Ref<Eigen::MatrixXi> F, Facetypes type)
{
    std::ifstream file(filename);
    std::string line;
    bool read_counts = false, read_vertices = false;
    int lineCount = 0, num_vertices{}, num_faces{}, num_edges, vertexCount = 0, faceCount = 0;

    while (std::getline(file, line))
    {
        // Skip comment, empty lines, property lines, and the 'PLY' header
        if (line.substr(0,7) == "comment" || line.empty() || (lineCount == 0 && line == "ply") || (line == "end_header") || (line.substr(0, 8) == "property"))
        {
            //cout << "skipping" << endl;
            continue;
        }
        // detect format    
        else if (line.substr(0,6) == "format")
        {
            // ascii
            if (line[7] == 'a'){
                cout << "Format: " << line << endl;
            }
            // binary
            else if (line[7]== 'b'){
                cout << "Format: " << line << endl;
            }
            continue;
        }
        else if (line.substr(0, 14) == "element vertex"){
            string num_vertex_string = line.substr(15, line.size()-15);
            num_vertices = std::stoi(num_vertex_string);
            cout << "element vertex " << num_vertices << endl;
            V.resize(num_vertices, 3);
            continue;
        }
        else if (line.substr(0, 12) == "element face"){
            string num_face_string = line.substr(13, line.size()-13);
            num_faces = std::stoi(num_face_string);
            cout << "element face " << num_faces << endl;
            F.resize(num_faces, 3);
            read_counts = true;
            continue;
        }


        // Separate line into tokens
        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};

        if (tokens.size() == 3)
        {
            if (read_vertices)
                throw std::runtime_error("Unexpected vertex line in faces section");

            if (vertexCount >= num_vertices)
                throw std::runtime_error("More vertices in file than expected!");
            
            // Read vertex coordinates
            V(vertexCount, 0) = std::stod(tokens[0]);
            V(vertexCount, 1) = std::stod(tokens[1]);
            V(vertexCount, 2) = std::stod(tokens[2]);
            vertexCount++;
            continue;
        }

        if (tokens.size() == 4)
        {
            if (!read_vertices)
                read_vertices = true;

            if (faceCount >= num_faces)
                throw std::runtime_error("More faces in file than expected!");

            if (!read_counts)
                throw std::runtime_error("File line counts missing!");
            
            if (std::stoi(tokens[0]) == 3) {
                // Read triangle
                F(faceCount, 0) = std::stoi(tokens[1]);
                F(faceCount, 1) = std::stoi(tokens[2]);
                F(faceCount, 2) = std::stoi(tokens[3]);
            } else if (std::stoi(tokens[0]) == 4) {
                // Read quads
                F(faceCount, 0) = std::stoi(tokens[1]);
                F(faceCount, 1) = std::stoi(tokens[2]);
                F(faceCount, 2) = std::stoi(tokens[3]);
                F(faceCount, 3) = std::stoi(tokens[4]);
            } else {
                throw std::runtime_error("Unsupported face type!");
            }

            faceCount++;
            continue;
        }

        throw std::runtime_error("Unexpected line in file!");
    }
}






void make_cube(igl::opengl::glfw::Viewer &viewer)
{
    // Inline mesh of a cube
    const Eigen::MatrixXd V = (Eigen::MatrixXd(8, 3) << 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0,
                               0.0, 1.0, 0.0,
                               0.0, 1.0, 1.0,
                               1.0, 0.0, 0.0,
                               1.0, 0.0, 1.0,
                               1.0, 1.0, 0.0,
                               1.0, 1.0, 1.0)
                                  .finished();
    const Eigen::MatrixXi F = (Eigen::MatrixXi(12, 3) << 1, 7, 5,
                               1, 3, 7,
                               1, 4, 3,
                               1, 2, 4,
                               3, 8, 7,
                               3, 4, 8,
                               5, 7, 8,
                               5, 8, 6,
                               1, 5, 6,
                               1, 6, 2,
                               2, 6, 8,
                               2, 8, 4)
                                  .finished()
                                  .array() -
                              1;
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
}




// Initialize the database and load in all meshes with their external features 
void init_database() {
    /* -- Load Labeled PSB dataset -- */

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Facetypes type;

    // Get all directories in the dataset
    for (const fs::directory_entry& dirEntry : fs::recursive_directory_iterator(LABELEDPSB_LOCATION)) {
        // Verify that dirEntry is a directory
        if (fs::is_directory(dirEntry)) {
            std::cout << "Entering class: " << dirEntry << std::endl;
            // Get all files in the directory
            for (const fs::directory_entry& fileEntry : fs::recursive_directory_iterator(dirEntry)) {
                // Verify that fileEntry is a file
                if (fs::is_regular_file(fileEntry)) {
                    // Get the file extension
                    std::string extension = fileEntry.path().extension().string();
                    // Verify that the file is a PLY file
                    if (extension == ".ply") {
                        // Load the mesh
                        load_PLY(fileEntry.path().string(), V, F, type);
                        // Add the mesh to the database
                        //database.push_back(V);
                        int num_vertices = V.rows();
                        int num_faces = F.rows();
                    }
                }
            }
        }
    } 
}



int main(int argc, char *argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::opengl::glfw::Viewer viewer;
    Facetypes type
    
    //make_cube(viewer);
    load_off("../data/LabeledDB_new/Airplane/61.off", V, F, type);
    //load_PLY(viewer, "m56.ply", type);

    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.core().align_camera_center(V, F);

    init_database();

    viewer.launch();

}

