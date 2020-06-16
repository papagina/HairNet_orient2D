//
// Created by zy on 17-1-22.
// Copyright (c) 2016 USC Super Meth Lab All rights reserved.
//
#pragma once



#include <iostream>
#include <stdarg.h>

#include <stdio.h>
//#include <windows.h>
#define GLEW_STATIC
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include "Vec.hpp"

#include <fstream>
#include <iostream>

// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Utils/getopt.h>
#include <opencv2/opencv.hpp>
#include "XForm.h"
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;

using namespace std;



class Renderer
{
private:

    MyMesh head_mesh;
    MyMesh body_mesh;


    GLfloat* g_vertex_buffer_data;
    GLfloat* g_color_buffer_data;



    bool first_time_bufferdata=true;



public:
    bool _render2Dorient=true;
    bool _add_noise_to_orient =false;
    int hair_vertex_buffer_size = 0;
    int hair_color_buffer_size = 0;

    int body_vertex_buffer_size=0;
    int body_color_buffer_size=0;

    void delete_buffer_data()
    {
        delete g_vertex_buffer_data;
        delete g_color_buffer_data;
    }

    void read_body_mesh()
    {
        if(body_mesh.faces_empty()==true)
        {
            body_mesh.request_vertex_normals();
            OpenMesh::IO::Options read_opt = OpenMesh::IO::Options::VertexNormal;

            //if( !OpenMesh::IO::read_mesh(mesh, "/home/yi/Documents/Hair/real_hair_data/torsoHair.obj", read_opt))
            if( !OpenMesh::IO::read_mesh(body_mesh, "../../model/female_halfbody_medium_in_realimage_coordinate.obj", read_opt))

            {
                cout<<"read body mesh error\n";
            }
            else
            {
                cout<<"read body mesh succeeded!\n";

            }



            cout<<"normals: "+to_string(body_mesh.has_vertex_normals())<<"\n";
            cout << "face: "+body_mesh.has_face_normals()<<"\n";
        }
    }



    void get_body_vertex_buffer(vector<GLfloat> &vertex_data, vector<GLfloat> &color_data)
    {

        //MyMesh mesh;

        read_body_mesh();



        OpenMesh::TriMesh_ArrayKernelT<>::FaceIter f_it(body_mesh.faces_begin()), f_end(body_mesh.faces_end());
        for (; f_it != f_end; ++f_it)
        {
            OpenMesh::TriMesh_ArrayKernelT<>::FaceVertexIter fv_it = body_mesh.fv_iter(f_it.handle());

            glBegin(GL_POLYGON);
            for (; fv_it; ++fv_it)
            {
                vertex_data.push_back(body_mesh.point(fv_it)[0]);
                vertex_data.push_back(body_mesh.point(fv_it)[1]);
                vertex_data.push_back(body_mesh.point(fv_it)[2]);
                color_data.push_back(1.0);
                color_data.push_back(1.0);
                color_data.push_back(1.0);

            }
            glEnd();

        }

        OpenMesh::TriMesh_ArrayKernelT<>::FaceIter f_it2(head_mesh.faces_begin()), f_end2(head_mesh.faces_end());
        for (; f_it2 != f_end2; ++f_it2)
        {
            OpenMesh::TriMesh_ArrayKernelT<>::FaceVertexIter fv_it = head_mesh.fv_iter(f_it2.handle());

            glBegin(GL_POLYGON);
            for (; fv_it; ++fv_it)
            {
                vertex_data.push_back(head_mesh.point(fv_it)[0]);
                vertex_data.push_back(head_mesh.point(fv_it)[1]);
                vertex_data.push_back(head_mesh.point(fv_it)[2]);
                color_data.push_back(1.0);
                color_data.push_back(1.0);
                color_data.push_back(1.0);

            }
            glEnd();

        }

    }


    void set_vertex_color_buffer(GLuint &vertexbuffer, GLuint &colorbuffer,  glm::mat4 MVP, CVec<3,float> camera_pos) {

        // Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
        // A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
        vector < GLfloat > body_vertex_data, body_color_data;

        get_body_vertex_buffer(body_vertex_data, body_color_data);


        body_vertex_buffer_size = body_vertex_data.size()/3;
        body_color_buffer_size = body_color_data.size()/3;

        vector < GLfloat > vertex_data, color_data;
        vertex_data = body_vertex_data;

        color_data = body_color_data;

        g_vertex_buffer_data = vertex_data.data();
        g_color_buffer_data = color_data.data();




        // This will identify our vertex buffer
        //GLuint vertexbuffer;
        // Generate 1 buffer, put the resulting identifier in vertexbuffer

        // The following commands will talk about our 'vertexbuffer' buffer
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        // Give our vertices to OpenGL.
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_data.size(), g_vertex_buffer_data, GL_STATIC_DRAW);




        // The following commands will talk about our 'vertexbuffer' buffer
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        // Give our vertices to OpenGL.
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * color_data.size(), g_color_buffer_data, GL_STATIC_DRAW);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void *) 0            // array buffer offset
        );


        // 2rst attribute buffer : vertices
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);

        glVertexAttribPointer(
                1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void *) 0            // array buffer offset
        );


    }




    //return MVP
    glm::mat4 set_camera(GLuint program_id,XForm<double> extrinsicMat)
    {

        glm::mat4 inMat_gl;
        glm::mat4 exMat_gl;


        // Projection matrix : 45�� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
        glm::mat4 Projection = glm::perspective(glm::radians(53.1301f), (float)(1.0/1.0), 0.01f, 1000.0f);

        // Camera matrix
        glm::mat4 View = glm::lookAt(
                glm::vec3(0,0,0), // Camera is at (4,3,3), in World Space
                glm::vec3(0,0,1), // and looks at the origin
                glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
        );

        inMat_gl=Projection*View;


        cout<<"inMat:\n";
        for(int i=0;i<4;i++) {
            for (int j = 0; j < 4; j++) {
                //inMat_gl[i][j] = intrinsicMat[j * 4 + i];
                cout<<inMat_gl[i][j]<<" ";

            }
            cout<<"\n";
        }

        //exMat_gl=glm::translate(glm::tvec3<double>(0,0,400));


        cout<<"exMat:\n";
        for(int i=0;i<4;i++) {
            for (int j = 0; j < 4; j++) {
                exMat_gl[i][j] = extrinsicMat[i * 4 + j];
                cout<<exMat_gl[i][j]<<" ";

            }
            cout<<"\n";
        }


        cout<<"mvp: \n";
        glm::mat4 mvp_gl=inMat_gl*exMat_gl;

        for(int i=0;i<4;i++) {
            for (int j = 0; j < 4; j++){
                cout << mvp_gl[i][j] << " ";
            }
            cout << "\n";
        }

        // Get a handle for our "MVP" uniform
        // Only during the initialisation
        GLuint MatrixID = glGetUniformLocation(program_id, "MVP");

        // Send our transformation to the currently bound shader, in the "MVP" uniform
        // This is done in the main loop since each model will have a different MVP matrix (At least for the M part)
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp_gl[0][0]);
    }

    GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path) {

        // Create the shaders
        GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
        GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

        // Read the Vertex Shader code from the file
        std::string VertexShaderCode;
        std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
        if (VertexShaderStream.is_open()) {
            std::string Line = "";
            while (getline(VertexShaderStream, Line))
                VertexShaderCode += "\n" + Line;
            VertexShaderStream.close();
        }
        else {
            printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
            getchar();
            return 0;
        }

        // Read the Fragment Shader code from the file
        std::string FragmentShaderCode;
        std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
        if (FragmentShaderStream.is_open()) {
            std::string Line = "";
            while (getline(FragmentShaderStream, Line))
                FragmentShaderCode += "\n" + Line;
            FragmentShaderStream.close();
        }

        GLint Result = GL_FALSE;
        int InfoLogLength;


        // Compile Vertex Shader
        printf("Compiling shader : %s\n", vertex_file_path);
        char const * VertexSourcePointer = VertexShaderCode.c_str();
        glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
        glCompileShader(VertexShaderID);

        // Check Vertex Shader
        glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
        glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
        if (InfoLogLength > 0) {
            std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
            glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
            printf("%s\n", &VertexShaderErrorMessage[0]);
        }



        // Compile Fragment Shader
        printf("Compiling shader : %s\n", fragment_file_path);
        char const * FragmentSourcePointer = FragmentShaderCode.c_str();
        glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
        glCompileShader(FragmentShaderID);

        // Check Fragment Shader
        glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
        glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
        if (InfoLogLength > 0) {
            std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
            glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
            printf("%s\n", &FragmentShaderErrorMessage[0]);
        }



        // Link the program
        printf("Linking program\n");
        GLuint ProgramID = glCreateProgram();
        glAttachShader(ProgramID, VertexShaderID);
        glAttachShader(ProgramID, FragmentShaderID);
        glLinkProgram(ProgramID);

        // Check the program
        glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
        glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
        if (InfoLogLength > 0) {
            std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
            glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
            printf("%s\n", &ProgramErrorMessage[0]);
        }


        glDetachShader(ProgramID, VertexShaderID);
        glDetachShader(ProgramID, FragmentShaderID);

        glDeleteShader(VertexShaderID);
        glDeleteShader(FragmentShaderID);

        return ProgramID;
    }

    void savecurrentImg(string filename, int width, int height)
    {

        float* pixels = new float[3 * width * height];

        glReadPixels(0.0, 0.0, width, height, GL_RGB, GL_FLOAT, pixels);
        cv::Mat renderImg_exr(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
        cv::Mat renderImg_png(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                float R = pixels[j * 3 + i*width * 3 + 0];
                float G = pixels[j * 3 + i*width * 3 + 1];
                float B = pixels[j * 3 + i*width * 3 + 2];

                renderImg_png.at<cv::Vec3b>(i, j)[0] = B*255;
                renderImg_png.at<cv::Vec3b>(i, j)[1] = G*255;
                renderImg_png.at<cv::Vec3b>(i, j)[2] = R*255;
            }
        }

        cv::Mat renderImg_png2;
        cv::flip(renderImg_png, renderImg_png2, 1);
        cv::imwrite(filename+".png", renderImg_png2);


        delete pixels;
    }


};
