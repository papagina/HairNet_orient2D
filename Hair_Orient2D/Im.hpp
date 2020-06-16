#ifndef IM_H
#define IM_H
/*
Szymon Rusinkiewicz
Princeton University

Im.h
This is smr's image-class-inna-.h
Convention: zero-based, upper-left origin, pixels at integers.

#define IM_PPMONLY before including this to get rid of the
external JPG and PNG dependencies.
*/

#include "Vec.hpp"
#include "Color.h"
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#ifndef IM_PPMONLY
# include <jpeglib.h>
# include <png.h>
#endif

#ifdef WIN32
# ifndef strcasecmp
#  define strcasecmp stricmp 
# endif
#endif

using namespace trimesh;
using namespace std;

class Im {
private:
	std::vector<Color> pix;

public:
	int w, h;
	bool is16bit;

	// Constructors
	Im() : w(0), h(0), is16bit(false)
		{}
	Im(int w_, int h_) : pix(w_*h_), w(w_), h(h_), is16bit(false)
		{}
	Im(int w_, int h_, const Color &col) : pix(w_*h_, col),
			w(w_), h(h_), is16bit(false)
		{}

	Im(int w_, int h_, float val) : pix(w_*h_, Color(val)),
			w(w_), h(h_), is16bit(false)
		{}
	Im(int w_, int h_, const float *data, bool gray=false) : pix(w_*h_),
			w(w_), h(h_), is16bit(false)
	{
		if (gray)
			for (int i=0; i < w*h; i++) pix[i] = Color(data[i]);
		else
			for (int i=0; i < w*h; i++) pix[i] = Color(&data[3*i]);
	}
	Im(int w_, int h_, const std::vector<float> &data, bool gray=false) :
			pix(w_*h_), w(w_), h(h_), is16bit(false)
	{
		if (gray)
			for (int i=0; i < w*h; i++) pix[i] = Color(data[i]);
		else
			for (int i=0; i < w*h; i++) pix[i] = Color(&data[3*i]);
	}

	// Using default copy constructor, assignment operator, destructor

	// Array access
	const Color &operator [] (int i) const
		{ return pix[i]; }
	Color &operator [] (int i)
		{ return pix[i]; }

	// Array access by row/column
	const Color &operator () (int x, int y) const
		{ // assert(x >= 0 && x < w && y >= 0 && y < h);
		  return pix[x + y * w]; }
	Color &operator () (int x, int y)
		{ return pix[x + y * w]; }

	// Interpolated access
	const Color lerp(float x, float y) const
	{
		x = clamp(x, 0.0f, w - 1.0f);
		y = clamp(y, 0.0f, h - 1.0f);
		int X = int(x), Y = int(y);
		float fx = x - float(X);
		float fy = y - float(Y);
		const Color &ll = pix[X + Y * w];
		const Color &lr = (fx > 0.0f) ? pix[X+1 + Y * w] : ll;
		const Color &ul = (fy > 0.0f) ? pix[X + (Y+1) * w] : ll;
		const Color &ur = (fx > 0.0f) ?
			((fy > 0.0f) ? pix[(X+1) + (Y+1) * w] : lr) :
			((fy > 0.0f) ? ul : ll);
		return (1.0f - fx) * ((1.0f - fy) * ll + fy * ul) +
		               fx  * ((1.0f - fy) * lr + fy * ur);
	}

	// Member operators
	Im &operator += (const Im &x)
	{
		assert(w == x.w);
		assert(h == x.h);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] += x[i];
		return *this;
	}
	Im &operator += (const Color &c)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] += c;
		return *this;
	}
	Im &operator += (const float x)
	{
		Color c(x);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] += c;
		return *this;
	}
	Im &operator -= (const Im &x)
	{
		assert(w == x.w);
		assert(h == x.h);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] -= x[i];
		return *this;
	}
	Im &operator -= (const Color &c)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] -= c;
		return *this;
	}
	Im &operator -= (const float x)
	{
		Color c(x);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] -= c;
		return *this;
	}
	Im &operator *= (const Im &x)
	{
		assert(w == x.w);
		assert(h == x.h);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] *= x[i];
		return *this;
	}
	Im &operator *= (const Color &c)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] *= c;
		return *this;
	}
	Im &operator *= (const float x)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] *= x;
		return *this;
	}
	Im &operator /= (const Im &x)
	{
		assert(w == x.w);
		assert(h == x.h);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] /= x[i];
		return *this;
	}
	Im &operator /= (const Color &c)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] /= c;
		return *this;
	}
	Im &operator /= (const float x)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i] /= x;
		return *this;
	}
	Im &min(const Im &x)
	{
		assert(w == x.w);
		assert(h == x.h);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i].min(x[i]);
		return *this;
	}
	Im &min(const Color &c)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i].min(c);
		return *this;
	}
	Im &min(const float x)
	{
		Color c(x);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i].min(c);
		return *this;
	}
	Im &max(const Im &x)
	{
		assert(w == x.w);
		assert(h == x.h);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i].min(x[i]);
		return *this;
	}
	Im &max(const Color &c)
	{
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i].max(c);
		return *this;
	}
	Im &max(const float x)
	{
		Color c(x);
		int n = w*h;
//#pragma omp parallel for
		for (int i = 0; i < n; i++)
			pix[i].max(c);
		return *this;
	}

	// Partial compatibility with vectors
	typedef Color value_type;
	typedef Color *pointer;
	typedef const Color *const_pointer;
	typedef Color *iterator;
	typedef const Color *const_iterator;
	typedef Color &reference;
	typedef const Color &const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;

	size_t size() const
		{ return w*h; }
	Color *begin()
		{ return &(pix[0]); }
	const Color *begin() const
		{ return &(pix[0]); }
	Color *end()
		{ return begin() + w*h; }
	const Color *end() const
		{ return begin() + w*h; }
	void clear()
		{ pix.clear(); w=h=0; is16bit = false; }
	bool empty() const
		{ return (w <= 0 || h <= 0); }
	void resize(int w_, int h_)
		{ w = w_; h = h_; pix.resize(w*h); }
	void resize(int w_, int h_, const Color &col)
		{ w = w_; h = h_; pix.resize(w*h, col); }
	void resize(int w_, int h_, float val)
		{ w = w_; h = h_; pix.resize(w*h, Color(val,val,val)); }
	const Color sum() const
		{ Color total; int n = w*h;
		  for (int i = 0; i < n; i++) total += pix[i];
		  return total; }
	const Color avg() const
		{ return sum() / float(w*h); }
	const Color min() const
		{ int n = w*h;
		  if (!n) return Color();
		  Color m = pix[0];
		  for (int i = 1; i < n; i++) m.min(pix[i]);
		  return m; }
	const Color max() const
		{ int n = w*h;
		  if (!n) return Color();
		  Color m = pix[0];
		  for (int i = 1; i < n; i++) m.max(pix[i]);
		  return m; }

	// Color transformations.  These are done in-place, unlike
	// the functions on Colors.
	void col_transform(float m11, float m12, float m13,
			   float m21, float m22, float m23,
			   float m31, float m32, float m33);
	void convert(Color::Colorspace src, Color::Colorspace dst);
	void gamma(float g), gamma(Color::Colorspace dst);
	void ungamma(float g), ungamma(Color::Colorspace dst);

	// Simple transformations
	void flipX(), flipY(), rotCW(), rotCCW();
	void crop(int start_x, int start_y, int new_w, int new_h);
	void set(int start_x, int start_y, int subimg_w, int subimg_h,
	         const Color &c);
	void set(const Color &c) { set(0, 0, w, h, c); }
	void set(int start_x, int start_y, int subimg_w, int subimg_h,
	         const Im &im2);
	void set(int start_x, int start_y, const Im &im2)
		{ set(start_x, start_y, im2.w, im2.h, im2); }

	// TOADD: convolution (direct, FFT-based), bilateral,
	// image scaling, general image transformations, stats?

	// Input/output
	bool read(const std::string &filename);
	bool write(const std::string &filename);

private:
	bool read_pbm(std::FILE *f);
	bool read_jpg(std::FILE *f);
	bool read_png(std::FILE *f);
	bool write_jpg(std::FILE *f);
	bool write_png(std::FILE *f);
	static inline bool we_are_little_endian()
		{ int tmp = 1;
		  return !!(* (unsigned char *) &tmp); }
};

// Nonmember operators
static inline const Im operator + (const Im &x, const Im &y)
{
	return Im(x) += y;
}
static inline const Im operator - (const Im &x, const Im &y)
{
	return Im(x) -= y;
}
static inline const Im operator * (const Im &x, const Im &y)
{
	return Im(x) *= y;
}
static inline const Im operator * (const Im &x, const float y)
{
	return Im(x) *= y;
}
static inline const Im operator * (const float x, const Im &y)
{
	return y * x;
}
static inline const Im operator * (const Im &x, const Color &c)
{
	return Im(x) *= c;
}
static inline const Im operator * (const Color &c, const Im &y)
{
	return y * c;
}
static inline const Im operator / (const Im &x, const Im &y)
{
	return Im(x) /= y;
}
static inline const Im operator / (const Im &x, const float y)
{
	return Im(x) /= y;
}
static inline const Im operator / (const float x, const Im &y)
{
	Im result(y);
	int n = result.w * result.h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		result[i] = x / result[i];
	return result;
}
static inline const Im operator / (const Im &x, const Color &c)
{
	return Im(x) /= c;
}
static inline const Im operator / (const Color &c, const Im &y)
{
	Im result(y);
	int n = result.w * result.h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		result[i] = c / result[i];
	return result;
}
static inline const Im &operator + (const Im &x)
{
	return x;
}
static inline const Im operator - (const Im &x)
{
	Im result(x);
	int n = result.w * result.h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		result[i] = -result[i];
	return result;
}
static inline bool operator ! (const Im &x)
{
	return x.empty();
}

// Other nonmember functions
static inline const Im min(const Im &x, const Im &y)
{
	return Im(x).min(y);
}
static inline const Im max(const Im &x, const Im &y)
{
	return Im(x).max(y);
}


// Color transformations
inline void Im::col_transform(float m11, float m12, float m13,
			      float m21, float m22, float m23,
			      float m31, float m32, float m33)
{
	int n = w*h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		pix[i] = pix[i].col_transform(m11,m12,m13,m21,m22,m23,m31,m32,m33);
}

inline void Im::convert(Color::Colorspace src, Color::Colorspace dst)
{
	int n = w*h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		pix[i] = pix[i].convert(src,dst);
}

inline void Im::gamma(float g)
{
	int n = w*h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		pix[i] = pix[i].gamma(g);
}

inline void Im::gamma(Color::Colorspace dst)
{
	int n = w*h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		pix[i] = pix[i].gamma(dst);
}

inline void Im::ungamma(float g)
{
	int n = w*h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		pix[i] = pix[i].ungamma(g);
}

inline void Im::ungamma(Color::Colorspace dst)
{
	int n = w*h;
//#pragma omp parallel for
	for (int i = 0; i < n; i++)
		pix[i] = pix[i].ungamma(dst);
}


// Geometric transformations
inline void Im::flipX()
{
	int w2 = w/2;
//#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		int row = y*w, rowend = row + w-1;
		for (int x = 0; x < w2; x++)
			std::swap(pix[row+x], pix[rowend-x]);
	}
}

inline void Im::flipY()
{
	int h2 = h/2;
//#pragma omp parallel for
	for (int y = 0; y < h2; y++) {
		int row = y*w, other = (h-1-y)*w;
		for (int x = 0; x < w; x++)
			std::swap(pix[x+row], pix[x+other]);
	}
}

inline void Im::rotCW()
{
	std::vector<Color> tmp(pix);
	std::swap(w,h);
//#pragma omp parallel for
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			pix[x+y*w] = tmp[y+(w-1-x)*h];
}

inline void Im::rotCCW()
{
	std::vector<Color> tmp(pix);
	std::swap(w,h);
//#pragma omp parallel for
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			pix[x+y*w] = tmp[h-1-y+x*h];
}

inline void Im::crop(int start_x, int start_y, int new_w, int new_h)
{
	std::vector<Color> tmp(pix);
	pix.resize(new_w * new_h);
//#pragma omp parallel for
	for (int y = 0; y < new_h; y++)
		for (int x = 0; x < new_w; x++)
			pix[x+y*new_w] = tmp[x+start_x+(y+start_y)*w];
	w = new_w;
	h = new_h;
}


// Set a subimage
inline void Im::set(int start_x, int start_y,
		    int subimg_w, int subimg_h,
		    const Color &c)
{
	if (start_x >= w || start_y >= h)
		return;
	if (subimg_w <= 0 || subimg_h <= 0)
		return;
	int end_x = start_x + subimg_w;
	int end_y = start_y + subimg_h;
	if (end_x < 0 || end_y < 0)
		return;
	if (start_x < 0) start_x = 0;
	if (start_y < 0) start_y = 0;
	if (end_x > w) end_x = w;
	if (end_y > h) end_y = h;
	subimg_w = end_x - start_x;
	subimg_h = end_y - start_y;

//#pragma omp parallel for
	for (int y = 0; y < subimg_h; y++)
		for (int x = 0; x < subimg_w; x++)
			pix[start_x+x+w*(start_y+y)] = c;
}

inline void Im::set(int start_x, int start_y,
		    int subimg_w, int subimg_h,
		    const Im &im2)
{
	if (start_x >= w || start_y >= h)
		return;
	if (subimg_w <= 0 || subimg_h <= 0)
		return;
	if (subimg_w > im2.w)
		subimg_w = im2.w;
	if (subimg_h > im2.h)
		subimg_h = im2.h;
	int end_x = start_x + subimg_w;
	int end_y = start_y + subimg_h;
	if (end_x < 0 || end_y < 0)
		return;
	if (start_x < 0) start_x = 0;
	if (start_y < 0) start_y = 0;
	if (end_x > w) end_x = w;
	if (end_y > h) end_y = h;
	subimg_w = end_x - start_x;
	subimg_h = end_y - start_y;

//#pragma omp parallel for
	for (int y = 0; y < subimg_h; y++)
		for (int x = 0; x < subimg_w; x++)
			pix[start_x+x+w*(start_y+y)] = im2(x,y);
}


// I/O
inline bool Im::read(const std::string &filename)
{
	using namespace std;

	FILE *f = strcmp(filename.c_str(), "-") ?
		fopen(filename.c_str(), "rb") : stdin;
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename.c_str());
		return false;
	}

	clear();

	// Parse magic number
	int m1 = fgetc(f), m2 = fgetc(f);
	ungetc(m2,f); ungetc(m1,f);

	if (m1 == 'P' && m2 == '4')
		return read_pbm(f);
	if (m1 == 0xff && m2 == 0xd8)
		return read_jpg(f);
	if (m1 == 0x89 && m2 == 'P')
		return read_png(f);

	bool is_pfm = false;
	int channels = 3;
	if (m1 == 'P' && m2 == '5')
		channels = 1;
	else if (m1 == 'P' && m2 == 'F')
		is_pfm = true;
	else if (m1 == 'P' && m2 == 'f')
		is_pfm = true, channels = 1;
	else if (!(m1 == 'P' && m2 == '6')) {
		fclose(f);
		fprintf(stderr, "Unknown file type\n");
		return false;
	}

	char buf[1024];
	fgets(buf, sizeof(buf), f);
	fgets(buf, sizeof(buf), f);
	while (buf[0] == '#')
		fgets(buf, sizeof(buf), f);

	// Get size
	if (sscanf(buf, "%d %d", &w, &h) != 2) {
		fclose(f);
		w = h = 0;
		fprintf(stderr, "Couldn't read dimensions\n");
		return false;
	}
	fgets(buf, sizeof(buf), f);
	while (buf[0] == '#')
		fgets(buf, sizeof(buf), f);

	// Get maxval
	bool need_swap = Im::we_are_little_endian();
	float maxval;
	if (is_pfm) {
		if (sscanf(buf, "%f", &maxval) != 1) {
			fclose(f);
			w = h = 0;
			fprintf(stderr, "Couldn't read maxval\n");
			return false;
		}
		if (maxval < 0.0f) {
			maxval = -maxval;
			need_swap = !need_swap;
		}
	} else {
		int m;
		if (sscanf(buf, "%d", &m) != 1 || m < 1 || m > 65535) {
			fclose(f);
			w = h = 0;
			fprintf(stderr, "Couldn't read maxval\n");
			return false;
		}
		maxval = m;
		is16bit = (m >= 256);
	}
	float scale = 1.0f / maxval;

	// Read data
	int n = w * h;
	int nbytes = is_pfm ? 4*channels*n : (1+int(is16bit))*channels*n;
	std::vector<unsigned char> data(nbytes);
	if (!fread(&data[0], nbytes, 1, f)) {
		fclose(f);
		w = h = 0;
		fprintf(stderr, "Couldn't read image pixels\n");
		return false;
	}

	fclose(f);
	pix.resize(n);

	if (is_pfm) {
		if (need_swap) {
//#pragma omp parallel for
			for (int i = 0; i < nbytes; i += 4) {
				std::swap(data[i  ], data[i+3]);
				std::swap(data[i+1], data[i+2]);
			}
		}
		const float *fdata = (const float *) &data[0];
		if (channels == 1) {
//#pragma omp parallel for
			for (int i = 0; i < n; i++)
				pix[i] = fdata[i];
		} else {
//#pragma omp parallel for
			for (int i = 0; i < n; i++)
				pix[i] = Color(&fdata[3*i]);
		}
	} else if (is16bit) {
		if (channels == 1) {
//#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				int p = 256*data[2*i] + data[2*i+1];
				pix[i] = scale * p;
			}
		} else {
//#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				int p1 = 256*data[6*i  ] + data[6*i+1];
				pix[i][0] = scale * p1;
				int p2 = 256*data[6*i+2] + data[6*i+3];
				pix[i][1] = scale * p2;
				int p3 = 256*data[6*i+4] + data[6*i+5];
				pix[i][2] = scale * p3;
			}
		}
	} else { // 8-bit
		if (channels == 1) {
//#pragma omp parallel for
			for (int i = 0; i < n; i++)
				pix[i] = scale * data[i];
		} else {
//#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				pix[i][0] = scale * data[3*i  ];
				pix[i][1] = scale * data[3*i+1];
				pix[i][2] = scale * data[3*i+2];
			}
		}
	}

	return true;
}

inline bool Im::read_pbm(std::FILE *f)
{
	using namespace std;

	char buf[1024];
	fgets(buf, sizeof(buf), f);
	fgets(buf, sizeof(buf), f);
	while (buf[0] == '#')
		fgets(buf, sizeof(buf), f);

	// Get size
	if (sscanf(buf, "%d %d", &w, &h) != 2) {
		fclose(f);
		w = h = 0;
		fprintf(stderr, "Couldn't read dimensions\n");
		return false;
	}

	// Read data
	pix.resize(w * h);
	int bytes_per_row = (w + 7) / 8;
	std::vector<unsigned char> data(bytes_per_row);
	int ind = 0;
	for (int i = 0; i < h; i++) {
		if (!fread(&data[0], bytes_per_row, 1, f)) {
			fclose(f);
			w = h = 0;
			pix.clear();
			fprintf(stderr, "Couldn't read image pixels\n");
			return false;
		}
		for (unsigned j = 0; j < w; j++) {
			unsigned byte = j >> 3u;
			unsigned bit = 7u - (j & 7u);
			if ((data[byte] >> bit) & 1u)
				pix[ind++] = Color::black();
			else
				pix[ind++] = Color::white();
		}
	}

	fclose(f);
	return true;
}

inline bool Im::read_jpg(std::FILE *f)
{
	using namespace std;

#ifdef IM_PPMONLY
	fclose(f);
	return false;
#else
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, f);
	jpeg_read_header(&cinfo, TRUE);
	cinfo.out_color_space = JCS_RGB;
	jpeg_start_decompress(&cinfo);
	w = cinfo.output_width;
	h = cinfo.output_height;
	pix.resize(w*h);
	std::vector<unsigned char> buf(3*w);
	for (int i = 0; i < h; i++) {
		JSAMPROW rowptr = (JSAMPROW) &buf[0];
		jpeg_read_scanlines(&cinfo, &rowptr, 1);
		for (int j = 0; j < w; j++)
			pix[w*i+j] = Color(buf[3*j], buf[3*j+1], buf[3*j+2]);
	}
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(f);
	return true;
#endif
}

inline bool Im::read_png(std::FILE *f)
{
	using namespace std;

#ifdef IM_PPMONLY
	fclose(f);
	return false;
#else
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
		0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_infop end_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, f);
	png_read_info(png_ptr, info_ptr);
	png_set_expand(png_ptr);
	png_set_strip_alpha(png_ptr);
	png_set_gray_to_rgb(png_ptr);
	w = png_get_image_width(png_ptr, info_ptr);
	h = png_get_image_height(png_ptr, info_ptr);
	/* w = info_ptr->width; */
	/* h = info_ptr->height; */
	int n = w*h;
	pix.resize(n);
	if (png_get_bit_depth(png_ptr, info_ptr) == 16) {
	/* if (info_ptr->bit_depth == 16) { */
		is16bit = true;
		if (Im::we_are_little_endian())
			png_set_swap(png_ptr);
		std::vector<unsigned short> buf(3*w*h);
		std::vector<png_bytep> row_pointers(h);
		for (int i = 0; i < h; i++)
			row_pointers[i] = (png_bytep) &buf[3*w*i];
		png_read_image(png_ptr, &row_pointers[0]);
		float scale = 1.0f / 65535;
		for (int i = 0; i < n; i++) {
			pix[i][0] = scale * buf[3*i];
			pix[i][1] = scale * buf[3*i+1];
			pix[i][2] = scale * buf[3*i+2];
		}
	} else {
		std::vector<unsigned char> buf(3*w*h);
		std::vector<png_bytep> row_pointers(h);
		for (int i = 0; i < h; i++)
			row_pointers[i] = (png_bytep) &buf[3*w*i];
		png_read_image(png_ptr, &row_pointers[0]);
		float scale = 1.0f / 255;
		for (int i = 0; i < n; i++) {
			pix[i][0] = scale * buf[3*i];
			pix[i][1] = scale * buf[3*i+1];
			pix[i][2] = scale * buf[3*i+2];
		}
	}
	png_read_end(png_ptr, end_ptr);
	png_destroy_read_struct(&png_ptr, &info_ptr, &end_ptr);
	fclose(f);
	return true;
#endif
}

inline bool Im::write(const std::string &filename)
{
	using namespace std;

	FILE *f = strcmp(filename.c_str(), "-") ?
		fopen(filename.c_str(), "wb") : stdout;
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename.c_str());
		return false;
	}

	const char *dot = strrchr(filename.c_str(), '.');
	if (dot && !strcasecmp(dot, ".jpg"))
		return write_jpg(f);
	else if (dot && !strcasecmp(dot, ".png"))
		return write_png(f);
	else if (dot && !strcasecmp(dot, ".pfm")) {
		fprintf(f, "PF\n%d %d\n%.1f\n", w, h,
			Im::we_are_little_endian() ? -1.0f : 1.0f);
		fwrite(&pix[0][0], 4*3*w*h, 1, f);
		fclose(f);
		return true;
	}
	// else write PPM

	fprintf(f, "P6\n%d %d\n%d\n", w, h, is16bit ? 65535 : 255);
	if (is16bit) {
		for (int i = 0; i < w*h; i++) {
			for (int j = 0; j < 3; j++) {
				float p = pix[i][j];
				unsigned short s = (unsigned short)
					clamp(int(round(p * 65535.0f)), 0, 65535);
				unsigned char c[2] = { static_cast<unsigned char>((s >> 8u)), static_cast<unsigned char>((s & 0xffu)) };
				fwrite(&c, 2, 1, f);
			}
		}
	} else {
		for (int i = 0; i < w*h; i++) {
			for (int j = 0; j < 3; j++) {
				float p = pix[i][j];
				unsigned char c = (unsigned char)
					clamp(int(round(p * 255.0f)), 0, 255);
				fwrite(&c, 1, 1, f);
			}
		}
	}
	fclose(f);
	return true;
}

inline bool Im::write_jpg(std::FILE *f)
{
	using namespace std;

#ifdef IM_PPMONLY
	fclose(f);
	return false;
#else
	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	cinfo.image_width = w;
	cinfo.image_height = h;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);  
	jpeg_set_quality(&cinfo, 90, TRUE);
	cinfo.optimize_coding = TRUE;
	cinfo.dct_method = JDCT_FLOAT;
	cinfo.comp_info[0].h_samp_factor = 1;
	cinfo.comp_info[0].v_samp_factor = 1;
	jpeg_stdio_dest(&cinfo, f);
	jpeg_start_compress(&cinfo, TRUE);
	std::vector<unsigned char> buf(3*w);
	JSAMPROW rowptr = (JSAMPROW) &buf[0];
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			const Color &p = pix[w*i+j];
			buf[3*j  ] = clamp(int(round(p[0] * 255.0f)), 0, 255);
			buf[3*j+1] = clamp(int(round(p[1] * 255.0f)), 0, 255);
			buf[3*j+2] = clamp(int(round(p[2] * 255.0f)), 0, 255);
		}
		jpeg_write_scanlines(&cinfo, &rowptr, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	fclose(f);
	return true;
#endif
}

inline bool Im::write_png(std::FILE *f)
{
	using namespace std;

#ifdef IM_PPMONLY
	fclose(f);
	return false;
#else
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
		0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, f);
	png_set_IHDR(png_ptr, info_ptr, w, h, is16bit ? 16 : 8,
		PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png_ptr, info_ptr);
	if (is16bit) {
		if (Im::we_are_little_endian())
			png_set_swap(png_ptr);
		std::vector<unsigned short> buf(3*w);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				const Color &p = pix[w*i+j];
				buf[3*j  ] = clamp(int(round(p[0] * 65535.0f)), 0, 65535);
				buf[3*j+1] = clamp(int(round(p[1] * 65535.0f)), 0, 65535);
				buf[3*j+2] = clamp(int(round(p[2] * 65535.0f)), 0, 65535);
			}
			png_write_row(png_ptr, (png_bytep) &buf[0]);
		}
	} else {
		std::vector<unsigned char> buf(3*w);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				const Color &p = pix[w*i+j];
				buf[3*j  ] = clamp(int(round(p[0] * 255.0f)), 0, 255);
				buf[3*j+1] = clamp(int(round(p[1] * 255.0f)), 0, 255);
				buf[3*j+2] = clamp(int(round(p[2] * 255.0f)), 0, 255);
			}
			png_write_row(png_ptr, (png_bytep) &buf[0]);
		}
	}
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(f);
	return true;
#endif
}

#endif
