/*
 * Linjie Luo - Princeton University
 * Chongyang Ma - Weta Digital
 */

#pragma once
#ifndef __ORIENT_MAP_HPP__
#define __ORIENT_MAP_HPP__

#include "Im.hpp"
#include "MathUtils.hpp"
#include <opencv2/opencv.hpp>

class OrientMap
{
public:
  OrientMap() : _image() {}

  OrientMap(int w, int h) : _image(w, h) {}

  OrientMap(const char* fileName)
  {
    read(fileName);
  }
    
    void read(Im& image)
    {
        _image = image;
        //for (int i = 0; i < _image.size(); i++) {
        //    vec c = _image[i] * 2.0f - vec(1, 1, 0);
        //    _image[i] = (c[2] > 0.0f ? c : vec());
        //}
        computeGradientMap();
    }
    
  const Im& get() const { return _image; }
  Im& get() { return _image; }

  Im getImage() const
  {
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      im[i] = vec();
      if (_image[i][2] > 0.0f) {
	im[i][0] = _image[i][0] * 0.5f + 0.5f; 
	im[i][1] = _image[i][1] * 0.5f + 0.5f;
	im[i][2] = 1.0f;
      }
    }
    return im;
  }

  inline void doubleAngleToAngle(
				 float cos2A, float sin2A, float& cosA, float& sinA) const
  {
    cosA = sqrt(0.5f + cos2A * 0.5f);
    sinA = sqrt(0.5f - cos2A * 0.5f) * sgn(sin2A);
  }

  inline void angleToDoubleAngle(
				 float cosA, float sinA, float& cos2A, float& sin2A) const
  {
    cos2A = sqr(cosA) - sqr(sinA);
    sin2A = 2.0f * sinA * cosA;
  }

    vec2 get_samples_value(std::vector<vec2> samples)
    {
        vec2 avg_o(0,0);
        for(int i=0;i<samples.size();i++)
        {
            float x = 0.5*(samples[i][0]-0.5);
            float y = samples[i][1];
            float xx = x*x-y*y;
            float yy = 2*x*y;

            avg_o=avg_o+vec2(xx,yy);
        }

        avg_o[0]=avg_o[0]/samples.size();
        avg_o[1]=avg_o[1]/samples.size();

        float x = avg_o[0]/sqrt(avg_o[0]*avg_o[0]+avg_o[1]*avg_o[1]);
        float y = avg_o[1]/sqrt(avg_o[0]*avg_o[0]+avg_o[1]*avg_o[1]);

        x=x+1.0;
        y=y+0.0;
        float norm = sqrt(x*x+y*y);
        x= x/norm;
        y=y/norm;

        if(y<0)
        {
            x=-x;
            y=-y;
        }

        return vec2(x*0.5+0.5, y);

    }

  void diffuse(float lowConfThresh = 0.01f, int numIters = 40) {
      float lowConf2 = sqr(lowConfThresh);
      cout << "\nLowConf: " << lowConfThresh << "\n";

      int diffuse_num = 0;
      for (int i = 0; i < _image.size(); i++) {
          vec & c = _image[i];
          if (c[2] < lowConfThresh)//if (c[2] > 0.0f && sqr(c[0]) + sqr(c[1]) < lowConf2)
          {
              c[2] = 0.5f; // mark as to be diffused
              diffuse_num = diffuse_num + 1;
          } else {
              c[2] = 1.0f;
          }
      }
      cout << "Diffuse number: " << diffuse_num << "\n";

      for (int iter = 0; iter <= numIters; iter++) {
          cout<<"diffuse iter: "<<iter<<"\n";
          int diffuse_num2=0;
          for (int y = 0; y < _image.h; y++)
              for (int x = 0; x < _image.w; x++) {

                  // high confident pixels
                  if (_image(x, y)[2] > 0.5f) {
                      continue;
                  }
                  // background pixels
                  if (_image(x, y)[2] <= 0.0f) {
                      continue;
                  }
                  // boundary pixels
                  if (x <= 0 || x >= _image.w - 1 ||
                      y <= 0 || y >= _image.h - 1)
                      continue;

                  /*
                  std::vector<vec2> samples;
                  samples.push_back(vec2(_image(x, y - 1)[0], _image(x, y - 1)[1]));
                  samples.push_back(vec2(_image(x-1, y )[0], _image(x-1, y )[1]));
                  samples.push_back(vec2(_image(x, y )[0], _image(x, y )[1]));
                  samples.push_back(vec2(_image(x, y +1)[0], _image(x, y +1)[1]));
                  samples.push_back(vec2(_image(x+1, y )[0], _image(x+1, y )[1]));

                  vec2 c = get_samples_value(samples);
                  */
                  vec c = (_image(x, y - 1) +
                           _image(x - 1, y) +
                           _image(x, y) +
                           _image(x + 1, y) +
                           _image(x, y + 1)) / 5.0f;
                  _image(x, y)[0] = c[0];
                  _image(x, y)[1] = c[1];
                  diffuse_num2++;

              }
          cout<<"diffuse_num: "<<diffuse_num2<<"\n";
      }
  }

  bool isForeground(int x, int y) const
  {
    if (0 <= x && x < _image.w && 0 <= y && y < _image.h &&
						_image(x, y)[2] > 0.0f) {
      return true;
    }
    return false;
  }

  bool read(const char* fileName)
  {
    if (!_image.read(fileName))
      return false;
    //for (int i = 0; i < _image.size(); i++) {
    //  vec c = _image[i] * 2.0f - vec(1, 1, 0);
   //   _image[i] = (c[2] > 0.0f ? c : vec());
    //}
    computeGradientMap();
    return true;
  }

    //added by Yi
    void write_orient_for_HairNet(const char* fileName) const
    {
        cv::Mat im(_image.h, _image.w, CV_32FC3);
        for (int i = 0; i < _image.size(); i++)
        {

            if (_image[i][2] > 0.0f) {

                //supose x is [0], y is [1]
                float x = _image[i][0];
                float y = _image[i][1];
                //cout<<y<<" ";
                im.at<cv::Vec3f>(i/_image.w, i% _image.w) [0] = x;//x*5.0;
                im.at<cv::Vec3f>(i/_image.w, i% _image.w) [1] = y;//y*0.5f+0.5f;
                im.at<cv::Vec3f>(i/_image.w, i% _image.w) [2] = 1.0f;

            }
        }
        cv::imwrite(fileName, im);
        /*
        Im im(_image.w, _image.h);
        for (int i = 0; i < _image.size(); i++)
        {
            im[i] = vec();
            if (_image[i][2] > 0.0f) {

                //supose x is [0], y is [1]
                float x = _image[i][0];
                float y = _image[i][1];
                //cout<<y<<" ";
                im[i][0] = x;//x*5.0;
                im[i][1] = y;//y*0.5f+0.5f;
                im[i][2] = 1.0f;

            }
        }
        im.write(fileName);
         */
    }
    void write(const char* fileName) const
    {
        Im im(_image.w, _image.h);
        for (int i = 0; i < _image.size(); i++) {
            im[i] = vec();
            if (_image[i][2] > 0.0f) {
                im[i][0] = _image[i][0] * 0.5f + 0.5f;
                im[i][1] = _image[i][1] * 0.5f + 0.5f;
                im[i][2] = 0.5f;
            }
        }
        im.write(fileName);
  }

  void writeOrient(const char* fileName) const
  {
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      im[i] = vec();
      if (_image[i][2] > 0.0f) {
	float conf = sqrt(sqr(_image[i][0]) + sqr(_image[i][1]));
	if (conf > 0.01f) {
	  im[i][0] = _image[i][0] / conf * 0.5f + 0.5f; 
	  im[i][1] = _image[i][1] / conf * 0.5f + 0.5f;
	  im[i][2] = 0.5f;
	} else {
	  im[i] = Color(0.5f);
	}
      }
    }
    im.write(fileName);
  }

  void writeHSV(const char* fileName) const
  {
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      const vec& c = _image[i];
      if (c[2] > 0.0f) {
	float conf = sqrt(sqr(c[0]) + sqr(c[1]));
	if (conf > 0.01f) {
	  float a = atan2(c[1] / conf, c[0] / conf);
	  im[i] = Color::hsv(a, conf, conf * 0.5f + 0.5f);
	} else {
	  im[i] = Color(0.5f);
	}
      }
    }
    im.write(fileName);
  }

  void writeConfidence(const char* fileName) const
  {
      cv::Mat im(_image.h, _image.w, CV_32FC3);
      for (int i = 0; i < _image.size(); i++) {

          float cof=0;
          if (_image[i][2] > 0.0f)
              cof = _image[i][2];//sqrt(sqr(_image[i][0]) + sqr(_image[i][1]));
          else
              cof = 0;

          //cout<<y<<" ";
          im.at<cv::Vec3f>(i / _image.w, i % _image.w) = cv::Vec3f(cof,cof,cof);


      }
      cv::imwrite(fileName, im);

      /*
    Im im(_image.w, _image.h);
    for (int i = 0; i < _image.size(); i++) {
      const vec& c = _image[i];
      if (c[2] > 0.0f)
	im[i] = Color(sqrt(sqr(c[0]) + sqr(c[1])));
      else
	im[i] = Color(0.0f);
    }
    im.write(fileName);
      */
  }

  void computeGradientMap()
  {
    int w = _image.w, h = _image.h;
    _grad.resize(w, h);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
	vec c0, c1;
	float l0, l1;
	c0 = _image(x - (x > 0), y);
	c1 = _image(x + (x<w-1), y);
	l0 = sqrt(sqr(c0[0]) + sqr(c0[1]));
	l1 = sqrt(sqr(c1[0]) + sqr(c1[1]));
	_grad(x, y)[0] =
	  (c0[2] > 0.0f && c1[2] > 0.0f ? (l1 - l0) * 0.5f : 0.0f);
	c0 = _image(x, y - (y > 0));
	c1 = _image(x, y + (y<h-1));
	l0 = sqrt(sqr(c0[0]) + sqr(c0[1]));
	l1 = sqrt(sqr(c1[0]) + sqr(c1[1]));
	_grad(x, y)[1] =
	  (c0[2] > 0.0f && c1[2] > 0.0f ? (l1 - l0) * 0.5f : 0.0f);
      }
  }

  // return orientation consistency energy
  inline float compare(
		       const vec2& point, const vec2& orient) const
  {
    vec2 thisOrient;
    float conf = sample(point, thisOrient, &orient);
    return min(dist(orient, thisOrient * conf), 1.0f);
  }


  inline float sample(
		      const vec2& point, vec2& orient, const vec2* refOrient = NULL) const
  {
    vec c = _image.lerp(point[0], point[1]);
    if (c[2] == 0.0f) return -1.0f;
    float conf = sqrt(sqr(c[0]) + sqr(c[1]));
    if (conf == 0.0f) return -1.0f;
    doubleAngleToAngle(c[0] / conf, c[1] / conf, orient[0], orient[1]);
    if (refOrient && (orient DOT *refOrient) < 0) orient = -orient;
    return conf;
  }

  inline vec2 sampleGradient(const vec2& point) const
  {
    vec g = _grad.lerp(point[0], point[1]);
    return vec2(g[0], g[1]);
  }

  void getRidgePoints(
		      vector<vec2>& points, float minConfidence, float minContrast) const
  {
    points.clear();
    Im nmax(_image.w, _image.h);
    for (int y = 0; y < _image.h; y++)
      for (int x = 0; x < _image.w; x++) {
	vec2 p(x, y);
	vec2 orient;
	float f1 = sample(p, orient);
	if (f1 < minConfidence)
	  continue;
	vec2 normal(orient[1], -orient[0]);
	float f0 = sample(p - normal, orient);
	float f2 = sample(p + normal, orient);
	float contrast = (f1 - max(f0, f2)) / f1;
	if (contrast > minContrast)
	  points.push_back(
			   snapToRidge(p - normal, f0, p, f1, p + normal, f2));
      }
  }

  inline vec2 snapToRidge(const vec2& point) const
  {
    float e0, e1, e2;
    vec2 orient;
    e1 = sample(point, orient);
    vec2 normal(orient[1], -orient[0]);
    e0 = sample(point - normal, orient);
    e2 = sample(point + normal, orient);
    return snapToRidge(point - normal, e0, point, e1, point + normal, e2);
  }

  inline vec2 snapToRidge(
			  const vec2& point0, float mag0,
			  const vec2& point1, float mag1,
			  const vec2& point2, float mag2) const
  {
    float E[3] = {mag0, mag1, mag2};
    float f, e;
    refineSubpixel(E, 3, 1, f, e);
    return (f < 1.0f ?
	    point0 * (1.0f - f) + point1 * f:
	    point1 * (2.0f - f) + point2 * (f - 1.0f));
  }

  int refineSubpixel(
		     const float* E, int n, int idep, 
		     float& depth, float& energy) const
  {
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
    int move = 0;
    while (idep > 0 && idep < n - 1) {
      e0 = E[idep-1]; e1 = E[idep]; e2 = E[idep+1];
      if (e0 < e1 && e1 < e2) {
	depth = float(idep) + 0.5f;
	energy = e1;
	move = 1;
	break;
	// e0 = e1; e1 = e2; 
	// e2 = E[min(idep+2, n-1)]; // shift right
	// idep++; move++;
      } else if (e0 > e1 && e1 > e2) {
	depth = float(idep) - 0.5f;
	energy = e1;
	move = -1;
	break;
	// e2 = e1; e1 = e0; 
	// e0 = E[max(idep-2, 0)]; // shift left
	// idep--; move--;
      } else {
	float d2E = e0 + e2 - 2.0f * e1;
	depth = (d2E != 0.0f ? idep + 0.5f * (e0 - e2) / d2E : idep);
	energy = e1;
	break;
      }
    }
    return move;
  }

protected:
  Im _image;
  Im _grad;
  vector<vec2> _ridge;
  //kdtree2f* _kd;
};


#if 0
#ifndef ORIENT_MAP_TESTER
#define ORIENT_MAP_TESTER    
#include "Random.hpp"

struct OrientMapTester {
  OrientMapTester()
  {
    Im im;
    im.read("lower5.png");
    OrientMap orientMap(im);
	
    vector<vec2> points1;
    vector<vec2> points2;
    while (points1.size() < 100) {
      vec2 p1;
      p1[0] = uniform(0.0f, float(im.w));
      p1[1] = uniform(0.0f, float(im.h));
      float alpha = uniform(-float(M_PI), float(M_PI));
      vec2 p2;
      p2[0] = p1[0] + 10.0f * cos(alpha);
      p2[1] = p1[1] + 10.0f * sin(alpha);
	    
      if (im(int(p1[0]), int(p1[1]))[2] > 0.0f &&
	  im(int(p2[0]), int(p2[1]))[2] > 0.0f) {
	points1.push_back(p1);
	points2.push_back(p2);
	// printf("%f %f, %f %f\n",
	//        p1[0], p1[1],
	//        p2[0], p2[1]);
      }
    }

    char fileName[1024];
    float dt = 1.0f;
    for (int step = 0; step < 50; step++) {
      Im temp = im;
      for (int i = 0; i < points1.size(); i++) {
	drawLine(temp, points1[i], points2[i], rand_color(i));
	vec2 p = (points2[i] + points1[i]) * 0.5f;
	vec2 o = (points2[i] - points1[i]) * 0.5f;
	vec2 gp, go;
	float f;
	orientMap.computeSnapGradient(
				      p, o, 4, 0.2, gp, go, f);
	vec2 d = (points2[i] - points1[i]);
	normalize(d);
	go -= d * (go DOT d);
	points1[i] -= (gp + go) * dt;
	points2[i] -= (gp - go) * dt;
	// printf("%f %f, %f %f\n",
	//        gp[0], gp[1], go[0], go[1]);
      }
      sprintf(fileName, "step%03d.png", step);
      temp.write(fileName);
    }

    // for (int i = 0; i < points1.size(); i++) {
    //     vec c = rand_color(i);
    //     drawLine(im, points1[i], points2[i], c);
    // }
    // im.write("lines.png");
  }

  static inline vec rand_color(int hash = -1)
  {
    static int id = 0;
    if (hash < 0) id++;
    else id = hash;
    return Color::hsv(-3.88 * id, 0.6 + 0.2 * sin(0.42 * id), 1.0);
  }

  void drawLine(Im& im, const vec2& p1, const vec2& p2, const vec& color)
  {
    static const int numSamples = 10;
    for (int i = 0; i <= numSamples; i++) {
      float alpha = float(i) / numSamples;
      vec2 p = p1 * (1 - alpha) + p2 * alpha;
      int x = round(p[0]);
      int y = round(p[1]);
      if (x >= 0 && x < im.w && y >= 0 && y < im.h)
	im(x, y) = color;
    }
  }

} orientMapTester;

#endif
#endif

#endif
