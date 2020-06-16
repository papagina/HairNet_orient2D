//
//  Orient2D.hpp
//  HairSketch
//
//  Created by Liwen on 12/20/14.
//  Copyright (c) 2014 Liwen Hu. All rights reserved.
//

#pragma once
#ifndef __ORIENT2D_HPP__
#define __ORIENT2D_HPP__

//#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <vector>
#include <string>
#include <fftw3.h>
#include "Im.hpp"
#include "OrientMap.hpp"

using namespace std;

#define USE_CHAI_CONFIDENCE 0

class COrient2D
{
public:
    COrient2D()
    {
        
    }
    
    COrient2D(const char* filename, const char* outfilename)
    {
        int ndegree = 180;
        double sigma_h = 0.5;
        double sigma_l = 1;
        double sigma_y = 4;
        int npass = 3;
        float lowConfThresh = 0.8f;
        int numIters = 40;
        
        Im buffer0, buffer1;
        buffer0.read(filename);
        Im mask;
        compute_mask(buffer0, sigma_l, mask);
        
        filter(buffer0, ndegree, sigma_h, sigma_l, sigma_y, buffer1);
        
        Im* buffers[] = { &buffer0, &buffer1 };
        int curr = 1;
        for (int i = 1; i < npass; i++) {
            float max_mag = 0.0f;
            for (int j = 0; j < buffers[curr]->size(); j++) {
                float mag = mask[j][0] != 0.0f ? (*buffers[curr])[j][1] : 0.0f;
                (*buffers[curr])[j][0] = mag;
                (*buffers[curr])[j][2] = mag;
                max_mag = max(max_mag, mag);
            }
            for (int j = 0; j < buffers[curr]->size(); j++) {
                (*buffers[curr])[j] /= max_mag;
            }
            filter(*buffers[curr], ndegree, sigma_h, sigma_l, sigma_y,
                   *buffers[1-curr]);
            curr = 1 - curr;
        }
        
        for (int j = 0; j < buffers[curr]->size(); j++)
            if (mask[j][0] != 0.0f)
                (*buffers[curr])[j][2] = 1.0f;
        
        viz_ori_2color(*buffers[curr]);
        m_orientMap.read(*buffers[curr]);
        m_orientMap.diffuse(lowConfThresh, numIters);
        m_orientMap.write_orient_for_HairNet(outfilename);
        //m_orientMap.writeConfidence(outfilename);
        //m_orientMap.write(outfilename);
    }
    
    ~COrient2D()
    {
        
    }
    
    void mexican_hat(
                     const fftw_complex *in, int w, int h, double sigma_xh,
                     double sigma_xl, double sigma_y, double theta, fftw_complex *out)
    {
        float s = (float)sin(theta), c = (float)cos(theta);
        double xhmult = -2.0 * sqr(M_PI * sigma_xh);
        double xlmult = -2.0 * sqr(M_PI * sigma_xl);
        double ymult = -2.0 * sqr(M_PI * sigma_y);
        
#pragma omp parallel for
        for (int y = 0; y < h; y++) {
            double ynorm = (y >= h/2) ? y - h : y;
            ynorm /= h;
            for (int x = 0; x <= w/2; x++) {
                double xnorm = (double) x / w;
                double xrot2 = sqr(s * xnorm - c * ynorm);
                double yrot2 = sqr(c * xnorm + s * ynorm);
                
                int i = x + y * (w/2+1);
                double g = exp(xhmult * xrot2 + ymult * yrot2) -
                exp(xlmult * xrot2 + ymult * yrot2);
                out[i][0] = in[i][0] * g;
                out[i][1] = in[i][1] * g;
            }
        }
    }

    // Filter FFT by Gaussian
    void gaussian(const fftw_complex *in, int w, int h, double sigma,
                  fftw_complex *out)
    {
        double mult = -2.0 * sqr(M_PI * sigma);
        
#pragma omp parallel for
        for (int y = 0; y < h; y++) {
            double ynorm = (y >= h/2) ? y - h : y;
            ynorm /= h;
            for (int x = 0; x <= w/2; x++) {
                double xnorm = (double) x / w;
                
                int i = x + y * (w/2+1);
                float g = (float)exp(mult * (xnorm*xnorm + ynorm*ynorm));
                out[i][0] = in[i][0] * g;
                out[i][1] = in[i][1] * g;
            }
        }
    }
    
    
    // Build a single level in the pyramid
    void filter_dense(const fftw_complex *imfft, int w, int h,
                      double sigma_h, double sigma_l, double sigma_y,
                      int orientations, Im &out)
    {
        out.clear();
        out.resize(w, h);
        int npix = w * h;
        
        fftw_complex *filtfft = (fftw_complex *)
        fftw_malloc(sizeof(fftw_complex) * (w/2+1) * h);
        double *filtered = (double *) fftw_malloc(sizeof(double) * w * h);
        fftw_plan idft = fftw_plan_dft_c2r_2d(
                                              h, w, filtfft, filtered, FFTW_ESTIMATE);
        
        Im m1(w, h), m2(w, h);
        for (int i = 0; i < orientations; i++) {
            double angle = M_PI * i / orientations;
            // mexican_hat(imfft, w, h, sigma_h, sigma_l, sigma_l, angle, filtfft);
            mexican_hat(imfft, w, h, sigma_h, sigma_l, sigma_y, angle, filtfft);
            fftw_execute(idft);
            
#if USE_CHAI_CONFIDENCE
            float c = cos(angle * 2.0f);
            float s = sin(angle * 2.0f);
            for (int j = 0; j < npix; j++) {
                float res = filtered[j];
                if (out[j][1] < res) {
                    out[j][0] = angle;
                    out[j][1] = res;
                }
                res = max(res, 0.0f);
                float res2 = res * res;
                m1[j][0] += res;
                m1[j][1] += res * c;
                m1[j][2] += res * s;
                m2[j][0] += res2;
                m2[j][1] += res2 * c;
                m2[j][2] += res2 * s;
            }
#else
            for (int j = 0; j < npix; j++) {
                float res = (float)filtered[j];
                if (fabs(out[j][1]) < fabs(res)) {
                    out[j][0] = (float)angle;
                    out[j][1] = res;
                }
            }
#endif
        }
        
#if USE_CHAI_CONFIDENCE
        // compute confidence
        for (int j = 0; j < npix; j++) {
            float a = out[j][0];
            float f = out[j][1];
            float c = cos(a * 2.0f);
            float s = sin(a * 2.0f);
            out[j][1] = ((m2[j][0] - 2.0f * m1[j][0] * f + f * f * orientations) -
                         (m2[j][1] - 2.0f * m1[j][1] * f) * c -
                         (m2[j][2] - 2.0f * m1[j][2] * f) * s);
        }
#endif
        
        fftw_destroy_plan(idft);
        fftw_free(filtered);
        fftw_free(filtfft);
    }
    
    void filter(
                const Im& im, int orientations,
                double sigma_h, double sigma_l, double sigma_y,  Im &out)
    {
        int w = im.w, h = im.h, npix = w * h;
        double *ffttmp = (double *) fftw_malloc(sizeof(double) * w * h);
        fftw_complex *imfft = (fftw_complex *)
        fftw_malloc(sizeof(fftw_complex) * (w/2+1) * h);
        
        Im curr_im(im);
        
        fftw_plan imdft = fftw_plan_dft_r2c_2d(h, w, ffttmp, imfft, FFTW_ESTIMATE);
        fftw_plan imidft = fftw_plan_dft_c2r_2d(h, w, imfft, ffttmp, FFTW_ESTIMATE);
        double fftscale = 1.0 / (w * h);
        for (int j = 0; j < npix; j++)
            ffttmp[j] = fftscale * curr_im[j].avg();
        fftw_execute(imdft);
        
        filter_dense(imfft, w, h, sigma_h, sigma_l, sigma_y, orientations, out);
        
        fftw_destroy_plan(imidft);
        fftw_destroy_plan(imdft);
        
        fftw_free(imfft);
        fftw_free(ffttmp);
    }
    
    void compute_mask(const Im& im, double sigma, Im& out)
    {
        const float fg_impact_thresh = 0.9999f;
        int w = im.w, h = im.h, npix = w * h;
        double *ffttmp = (double *) fftw_malloc(sizeof(double) * w * h);
        fftw_complex *imfft = (fftw_complex *)
        fftw_malloc(sizeof(fftw_complex) * (w/2+1) * h);
        
        fftw_plan imdft = fftw_plan_dft_r2c_2d(h, w, ffttmp, imfft, FFTW_ESTIMATE);
        fftw_plan imidft = fftw_plan_dft_c2r_2d(h, w, imfft, ffttmp, FFTW_ESTIMATE);
        
        double fftscale = 1.0 / (w * h);
        for (int j = 0; j < npix; j++)
            ffttmp[j] = im[j].sum() ? fftscale : 0.0;
        
        fftw_execute(imdft);
        gaussian(imfft, w, h, sigma, imfft);
        fftw_execute(imidft);
        
        out.resize(w, h);
        for (int j = 0; j < npix; j++) {
            out[j] = ffttmp[j] >= fg_impact_thresh ? Color(1.0) : Color(0.0);
        }
        
        fftw_destroy_plan(imidft);
        fftw_destroy_plan(imdft);
        
        fftw_free(imfft);
        fftw_free(ffttmp);
    }
    
    void viz_ori_2color(Im &im)
    {
        int npix = (int)im.size();
        float max_mag = 0.0f;
        for (int i = 0; i < npix; i++)
            if (im[i][2] > 0.0f)
                max_mag = max(max_mag, im[i][1]);
        vector<float> mags;
        mags.reserve(npix);
        for (int i = 0; i < npix; i++) {
            if (im[i][2] > 0.0f && im[i][1] > 0.0f) {
                mags.push_back(im[i][1]);
            }
        }
        
        nth_element(mags.begin(), mags.begin() + mags.size() / 2, mags.end());
        float scale = 0.5f / mags[mags.size() / 2];
        // float scale = 1.0f / max_mag;
        
        for (int i = 0; i < npix; i++) {
            if (im[i][2] == 0.0f)
                im[i] = Color::black();
            else if (im[i][1] <= 0.0f) {
                im[i] = Color(0.5f);
            } else {
                im[i] = Color(
                              -cos( im[i][0])  * 0.5f + 0.5f,
                              sin( im[i][0]),
                              im[i][2]);
                /*
                im[i] = Color(
                        cos(2.0f * im[i][0]) * im[i][1] * scale * 0.5f + 0.5f,
                        sin(2.0f * im[i][0]) * im[i][1] * scale * 0.5f + 0.5f,
                        0.5f);
                        */
            }
        }
    }
    
private:
    OrientMap m_orientMap;
};

#endif
