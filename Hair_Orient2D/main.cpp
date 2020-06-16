//
//  main.cpp
//
//  Created by Yi on 06/28/17.
//  Copyright (c) 2017 Yi@USC. All rights reserved.
//



#include <iostream>
#include <stdarg.h>

#include "Orient2D.hpp"

#include "Renderer.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#include <thread>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sys/stat.h>

using namespace std;




string intToStrLen5(int i)
{
    string s;
    char t[256];
    snprintf(t, 10, "%05d",i);
    //sprintf_s(t, "%05d", i);
    s = t;
    return s;
}



void visualize_and_save_body_image(string exMat_folder,  string out_body_img_folder)
{


    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);


    GLFWwindow* window = glfwCreateWindow(800, 800, "OpenGL", nullptr, nullptr); // Windowed


    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    glewInit();

    //VAO
    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);


    // Create and compile our GLSL program from the shaders
    Renderer renderer;
    renderer._render2Dorient=true;

    renderer._add_noise_to_orient=true;
    GLuint programID = renderer.LoadShaders("SimpleVertexShader.vertexshader", "SimpleFragmentShader.fragmentshader");

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Use our shader
    glUseProgram(programID);

    //vector<Vec3d> eye_pos_lst;
    //eye_pos_lst.push_back(Vec3d(0,1.5,2.5));
    //eye_pos_lst.push_back(Vec3d(2.5,1.5,0));
    //eye_pos_lst.push_back(Vec3d(0,1.5,-2.5));
    //eye_pos_lst.push_back(Vec3d(-2.5,1.5,0));

    int hair_num=0;

    DIR           *dirp;
    struct dirent *directory;

    dirp = opendir(exMat_folder.data());
    if (!dirp) {
        cout<<"no such hair folder.\n";
        return;

    }
    GLuint vertexbuffer, colorbuffer;
    glGenBuffers(1, &vertexbuffer);
    glGenBuffers(1, &colorbuffer);
    while ((directory = readdir(dirp)) != NULL) {
        string d_name = directory->d_name;


        if (d_name.length() < 5)
            continue;

        string exMat_fn = exMat_folder + directory->d_name;
        string input_name0 = directory->d_name;
        string input_name = input_name0.substr(0, input_name0.size() - 4);

        XForm<double> exMat;
        exMat.read(exMat_fn);
        //std::ifstream fin(hair_fn);

        cout << "image name: " + input_name << "\n";

        // Enable depth test
        glEnable(GL_DEPTH_TEST);
        // Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS);


        glfwSwapBuffers(window);

        float eye_pos_x =0;
        float eye_pos_y = 0;
        float eye_pos_z =0;

        CVec<3,float> eye_pos = CVec<3,float>(0,0,0);

        CVec<3,float> look_pos = CVec<3,float>(0,0,1);

        CVec<3,float> head_up = CVec<3,float>(0,1,0);

        glm::mat4 MVP = renderer.set_camera(programID, exMat);

        renderer.set_vertex_color_buffer(vertexbuffer, colorbuffer, MVP, eye_pos);






        // Draw the triangle !
        //glDrawArrays(GL_TRIANGLES, 0, renderer.body_vertex_buffer_size);

        glDrawArrays(GL_LINES, 0,
                     renderer.hair_vertex_buffer_size); // Starting from vertex 0; 3 vertices total -> 1 triangle

        glDrawArrays(GL_TRIANGLES, renderer.hair_vertex_buffer_size,
                     renderer.hair_vertex_buffer_size + renderer.body_vertex_buffer_size);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);


        glfwPollEvents();

        string body_img_fn =out_body_img_folder + input_name ;
        renderer.savecurrentImg(body_img_fn, 800, 800);

        //glClearBufferData(GL_ARRAY_BUFFER,GL_RGB,GL_RGB,GL_FLOAT,&vertexbuffer);
        //glClearBufferData(GL_ARRAY_BUFFER,GL_RGB,GL_RGB,GL_FLOAT,&colorbuffer);


        //renderer.delete_buffer_data();


    }


    glfwTerminate();
}

void save_orient_with_body_image(string orient_image_fn, string body_image_fn, string seg_image_fn, string out_orient_with_body_image_fn)
{
    cv::Mat orient_img=cv::imread(orient_image_fn, cv::IMREAD_UNCHANGED);
    cv::Mat body_img = cv::imread(body_image_fn, cv::IMREAD_GRAYSCALE);
    cv::Mat seg_img = cv::imread(seg_image_fn, cv::IMREAD_GRAYSCALE);
    cv::Mat orient_with_body_png(orient_img.rows, orient_img.cols, CV_8UC3);
    cv::Mat orient_with_body_exr(orient_img.rows, orient_img.cols, CV_32FC3);

    for (int i=0;i<orient_img.cols;i++)
        for (int j=0;j<orient_img.rows;j++)
        {
            int is_body= body_img.at<uchar>(i,j);
            int is_hair = seg_img.at<uchar>(i,j);
            orient_with_body_exr.at<cv::Vec3f>(i,j)[0]=orient_img.at<cv::Vec3f>(i,j)[0];
            orient_with_body_exr.at<cv::Vec3f>(i,j)[1]=orient_img.at<cv::Vec3f>(i,j)[1];
            orient_with_body_exr.at<cv::Vec3f>(i,j)[2]=1.0;

            if(is_hair>122)
            {
                orient_with_body_exr.at<cv::Vec3f>(i,j)[2]=1.0;
            }
            if( (is_body>122) && (is_hair<122))
            {
                orient_with_body_exr.at<cv::Vec3f>(i,j)=cv::Vec3f(0,0,0.5);
            }
            if( (is_body<122) && (is_hair<122))
                orient_with_body_exr.at<cv::Vec3f>(i,j)=cv::Vec3f(0,0,0);

        }

    cv::imwrite(out_orient_with_body_image_fn+".exr", orient_with_body_exr);

    cv::Mat orient_with_body_exr2(orient_img.rows, orient_img.cols, CV_32FC3);
    orient_with_body_exr2=orient_with_body_exr*255;
    orient_with_body_exr2.convertTo(orient_with_body_png, CV_8UC3);
    cv::imwrite(out_orient_with_body_image_fn+".png", orient_with_body_png);


}

void save_orient_with_body_image_from_folder(
       string body_img_folder,
        string input_img_folder, string output_orient_img_folder, string hair_seg_folder, string output_orient_img_with_body_folder)
{
    DIR           *dirp;
    struct dirent *directory;

    dirp = opendir(input_img_folder.data());
    if (!dirp) {
        cout<<"no this input_img folder.\n";
        return;
    }

    while ((directory = readdir(dirp)) != NULL) {
        string d_name = directory->d_name;


        if (d_name.length() < 5)
            continue;
        if((d_name.substr(d_name.size()-4, d_name.size())!=".png")&&(d_name.substr(d_name.size()-4, d_name.size())!=".jpg"))
            continue;

        string input_img_fn = input_img_folder + directory->d_name;
        string hair_name0 = directory->d_name;
        string hair_name = hair_name0.substr(0, hair_name0.size() - 4);
        cout<<"Process "<<input_img_fn<<"\n";

        string body_img_fn = body_img_folder+hair_name+".png";



        string orient_img_fn=output_orient_img_folder+hair_name+".exr";

        std::ifstream infile(orient_img_fn);
        if(!infile.good())
            COrient2D orient2D (input_img_fn.data(),orient_img_fn.data());

        string hairseg_fn=hair_seg_folder+hair_name+".png";
        string orient_img_with_body_fn=output_orient_img_with_body_folder+hair_name;

        cout<<"save orient with body image.\n";
        cout<<hairseg_fn<<"\n";
        cout<<orient_img_with_body_fn<<"\n";
        cout<<body_img_fn<<"\n";
        cout<<orient_img_fn<<"\n";
        save_orient_with_body_image(orient_img_fn,body_img_fn,hairseg_fn,orient_img_with_body_fn);

    }
}

cv::Vec3f get_sample(std::vector<cv::Vec3f> samples, std::vector<float> weights)
{
    int body_num = 0;
    int hair_num=0;
    int bg_num=0;

    for(int i=0;i<samples.size();i++)
    {
        if(samples[i][2]==1.0)
            hair_num++;
        if(samples[i][2]==0.5)
            body_num++;
        if(samples[i][2]==0)
            bg_num++;
    }

    if((bg_num>=body_num) &&(bg_num>=hair_num))
        return cv::Vec3f(0,0,0);
    if((body_num>=bg_num)&&(body_num>=hair_num))
        return cv::Vec3f(0,0,0.5);

    //calculate avg
    cv::Vec2f avg_o(0,0);
    int weights_sum=0;
    for(int i=0;i<samples.size();i++)
    {
        if(samples[i][2]==1.0)
        {
            if( (samples[i][0]>1.0) || (samples[i][0]<0)||(samples[i][1]>1.0) || (samples[i][1]<0) )
                cout<<"wrong orient: "<<samples[i][0]<<" "<<samples[i][1]<<"\n";

            float x= 2.0*(samples[i][0]-0.5);
            float y=samples[i][1];

            float norm = sqrt(x*x+y*y);
            if (norm==0)
                continue;
            x=x/norm;
            y=y/norm;

            //cout<<"gt: "<<x<<" "<<y<<"\n";
            avg_o[0]=avg_o[0]+(x*x-y*y)*weights[i];
            avg_o[1]=avg_o[1]+(2*x*y)*weights[i];
            weights_sum=weights[i]+weights_sum;
        }
    }

    if(weights_sum==0)
        return cv::Vec3f(0,0,0.5);

    avg_o[0]=avg_o[0]/weights_sum;
    avg_o[1]=avg_o[1]/weights_sum;


    float norm = sqrt( avg_o[0]*avg_o[0] +avg_o[1]*avg_o[1] );
    if(norm==0)
        return cv::Vec3f(0,0,0.5);
    avg_o[0]= avg_o[0]/ norm;
    avg_o[1]= avg_o[1]/ norm;

    //cout<<"af: "<<avg_o[0]<<" "<<avg_o[1]<<"\n";
    avg_o = avg_o+cv::Vec2f(1,0);

    norm = sqrt( avg_o[0]*avg_o[0] +avg_o[1]*avg_o[1] );
    if(norm==0)
        return cv::Vec3f(0,0,0.5);
    avg_o[0]= avg_o[0]/ norm;
    avg_o[1]= avg_o[1]/ norm;


    if(avg_o[1]<0) {
        avg_o[1] = avg_o[1] * -1.0;
        avg_o[0]=avg_o[0]*-1.0;
    }

    //cout<<"af: "<<avg_o[0]<<" "<<avg_o[1]<<"\n";
    return cv::Vec3f(avg_o[0]*0.5+0.5, avg_o[1], 1.0);

    float min_dis = 12345;
    cv::Vec2f min_o(0,0);
    for(int i=0;i<samples.size();i++)
    {
        if(samples[i][2]==1.0)
        {
            cv::Vec2f dis_v = cv::Vec2f(samples[i][0], samples[i][1]) - avg_o;
            double dis = (dis_v[0]*dis_v[0] *dis_v[1]*dis_v[1]);
            if(dis<min_dis)
            {
                min_dis=dis;
                min_o=cv::Vec2f(samples[i][0], samples[i][1]);
            }
        }
    }
    return cv::Vec3f(avg_o[0], avg_o[1], 1.0);


}

cv::Mat resize_orient_img(cv::Mat src, int tar_cols, int tar_rows)
{
    double scale_cols = double(src.cols)/double(tar_cols);
    double scale_rows= double(src.rows)/double(tar_rows);
    double radius=2;

    cv::Mat tar(tar_cols, tar_rows, CV_32FC3);

    for(int i = 0; i<tar_rows;i++)
    {
        for(int j=0;j<tar.cols;j++)
        {
            //process orient
            std::vector<cv::Vec3f> samples;
            std::vector<float> weights;
            int center_r = i*scale_rows;
            int center_c = j*scale_cols;
            int top_r = center_r-radius;
            int down_r=center_r+radius;
            if(top_r<0)
                top_r=0;
            if(down_r>src.rows)
                down_r=src.rows;

            int left_c = center_c-radius;
            int right_c=center_c+radius;
            if(left_c<0)
                left_c=0;
            if(right_c>src.cols)
                right_c=src.cols;

            for(int ii=top_r; ii<=down_r;ii++)
                for(int jj=left_c;jj<=right_c;jj++) {
                    float w = 1/(abs( ii-double(center_r) )+abs( jj-double(center_c) )+1.0);
                    samples.push_back(src.at<cv::Vec3f>(jj, ii));
                    weights.push_back(w);
                }

            tar.at<cv::Vec3f>(j, i)=get_sample(samples, weights);

        }
    }
    return tar;
}

void resize_orient_imgs(string src_folder, string tar_folder, int cols, int rows)
{
    DIR           *dirp;
    struct dirent *directory;

    dirp = opendir(src_folder.data());
    if (!dirp) {
        cout<<"no this input_img folder.\n";
        return;
    }

    while ((directory = readdir(dirp)) != NULL) {
        string d_name = directory->d_name;

        if (d_name.length() < 5)
            continue;
        if( (d_name.find(".exr")==std::string::npos) || (d_name[d_name.length()-1]!='r'))
            continue;

        cout<<"process "<<d_name<<"\n";

        string src_fn = src_folder + d_name;
        cv::Mat src_img = cv::imread(src_fn, cv::IMREAD_UNCHANGED);
        cv::Mat tar_img = resize_orient_img(src_img, cols, rows);

        cv::imwrite(tar_folder+d_name, tar_img);
        cv::imwrite(tar_folder+d_name+".png", tar_img*255);

    }
}


//rotation is xzy translation is xyz
void save_transformation(glm::vec3 translation, glm::vec3 rotation, string fn)
{
    string out_s="";
    out_s=out_s+to_string(translation[0])+" "+to_string(translation[1])+" "+to_string(translation[2])
          +" " +to_string(rotation[0])+" "+to_string(rotation[1])+" "+to_string(rotation[2]);
    ofstream out(fn);
    out<<out_s;
    out.close();

}
void save_body_transformation_in_sim_coordinate(string exMat_folder, string real2sim_mat_fn, string out_transformation_folder)
{
    DIR           *dirp;
    struct dirent *directory;

    dirp = opendir(exMat_folder.data());
    if (!dirp) {
        cout << "no this exMat folder.\n";

    }



    while ((directory = readdir(dirp)) != NULL) {



        string d_name = directory->d_name;

        if(d_name.size()<=4)
            continue;
        string exMat_fn = exMat_folder+d_name;
        string out_transformation_fn=out_transformation_folder+d_name;

        cout<<exMat_fn<<"\n";

        XForm<double> exMat;
        exMat.read(exMat_fn);

        glm::mat4 exMat_gl;
        for(int i=0;i<4;i++) {
            for (int j = 0; j < 4; j++) {
                exMat_gl[j][i] = exMat[i * 4 + j];
            }
        }




        glm::mat3 rotation_gl;
        for(int i=0;i<3;i++) {
            for (int j = 0; j < 3; j++) {
                rotation_gl[i][j] = exMat_gl[i][j];
                cout<<rotation_gl[i][j]<<" ";
            }
            cout<<"\n";
        }

        glm::quat q=glm::quat_cast(rotation_gl);
        glm::vec3 euler = glm::eulerAngles(q);

        glm::vec3 rotation = glm::vec3(euler[0],euler[2],euler[1]);


        glm::vec3 translate = glm::vec3(exMat_gl[0][3]*-0.00124723, exMat_gl[1][3]*-0.00124723, (exMat_gl[2][3]*-0.00124723+0.6));

        cout<<"translate: \n";
        cout<<translate[0]<<" "<<translate[1]<<" "<<translate[2]<<"\n";

        save_transformation(translate, rotation, out_transformation_fn);

    }

}


int main(int argc, char**argv) {

    if (argc < 3) {
        cout<<"Need three argv. bool has_body_imgs; string hair_folder;";
        return 1;
    }
    bool has_body_imgs = false;
    if (argv[1][0] == '1') {
        has_body_imgs = true;
    }

    string hair_folder = argv[2];

    string img_folder = hair_folder + "img/";
    string hair_seg_folder = hair_folder + "seg/";

    string out_body_img_folder = hair_folder + "body_img/";


    string out_orient_img_folder = hair_folder + "orient_img/";
    mkdir(out_orient_img_folder.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string out_orient_img_with_body_folder = hair_folder + "orient_img_with_body/";
    mkdir(out_orient_img_with_body_folder.data(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string out_orient_img_with_body_256_folder = hair_folder + "orient_img_with_body_256/";
    mkdir(out_orient_img_with_body_256_folder.data(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


    if (has_body_imgs == false) {

        cout << "Render body images.\n";
        string exMat_folder = hair_folder + "exMat/";
        mkdir(out_body_img_folder.data(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        visualize_and_save_body_image(exMat_folder, out_body_img_folder);
        save_body_transformation_in_sim_coordinate(exMat_folder,
                                                   "../../model/real2synthetic_head_transformation.txt",
                                                   out_orient_img_with_body_256_folder);
    }

    cout << "compute orientation map in original size (800*800).\n";
    save_orient_with_body_image_from_folder(out_body_img_folder, img_folder, out_orient_img_folder, hair_seg_folder,
                                            out_orient_img_with_body_folder);

    cout << "Resize orient imgs to 256*256.\n";
    resize_orient_imgs(out_orient_img_with_body_folder,
                       out_orient_img_with_body_256_folder,
                       256, 256);






    return 0;
}
