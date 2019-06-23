#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <math.h>
#include <sstream>
#include <string>

#define NORMALS


using std::cerr;
using std::endl;




vtkImageData *
NewImage(int width, int height)
{
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

    return img;
}

void
WriteImage(vtkImageData *img, const char *filename)
{
   std::string full_filename = filename;
   full_filename += ".png";
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}





struct LightingParameters
{
    LightingParameters(void)
    {
         lightDir[0] = -0.6;
         lightDir[1] = 0;
         lightDir[2] = -0.8;
         Ka = 0.3;
         Kd = 0.7;
         Ks = 2.3;
         alpha = 2.5;
    };
  

    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
};

LightingParameters lp;




class Matrix
{
  public:
    double          A[4][4];

    void            TransformPoint(const double *ptIn, double *ptOut);
    static Matrix   ComposeMatrices(const Matrix &, const Matrix &);
    void            Print(ostream &o);
};

void
Matrix::Print(ostream &o)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        char str[256];
        sprintf(str, "(%.7f %.7f %.7f %.7f)\n", A[i][0], A[i][1], A[i][2], A[i][3]);
        o << str;
    }
}

Matrix
Matrix::ComposeMatrices(const Matrix &M1, const Matrix &M2)
{
    Matrix rv;
    for (int i = 0 ; i < 4 ; i++)
        for (int j = 0 ; j < 4 ; j++)
        {
            rv.A[i][j] = 0;
            for (int k = 0 ; k < 4 ; k++)
                rv.A[i][j] += M1.A[i][k]*M2.A[k][j];
        }

    return rv;
}

void
Matrix::TransformPoint(const double *ptIn, double *ptOut)
{
    ptOut[0] = ptIn[0]*A[0][0]
             + ptIn[1]*A[1][0]
             + ptIn[2]*A[2][0]
             + ptIn[3]*A[3][0];
    ptOut[1] = ptIn[0]*A[0][1]
             + ptIn[1]*A[1][1]
             + ptIn[2]*A[2][1]
             + ptIn[3]*A[3][1];
    ptOut[2] = ptIn[0]*A[0][2]
             + ptIn[1]*A[1][2]
             + ptIn[2]*A[2][2]
             + ptIn[3]*A[3][2];
    ptOut[3] = ptIn[0]*A[0][3]
             + ptIn[1]*A[1][3]
             + ptIn[2]*A[2][3]
             + ptIn[3]*A[3][3];
}




class Camera
{
  public:
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];

};






double SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

Camera
GetCamera(int frame, int nframes)
{
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0;
    c.focus[1] = 0;
    c.focus[2] = 0;
    c.up[0] = 0;
    c.up[1] = 1;
    c.up[2] = 0;
    return c;
}





class Triangle
{
  public:
      double         X[3];
      double         Y[3];
      double	     Z[3];
      double 	     colors[3][3];
      double 		 normals[3][3];
      double 		 shading[3];

  // would some methods for transforming the triangle in place be helpful?
};

class Screen
{
  public:
      unsigned char   *buffer;
      double *zbuffer;
      int width, height;
	void SetPixel(int r, int c, double col[3], double *zbuffer, double newZval, double shading);
  // would some methods for accessing and setting pixels be helpful?
};

//added a SetPixel method as recommended in class
	void Screen::SetPixel(int r, int c, double col[3], double *zbuffer, double newZval, double shading){
	if (r < 0 || r >=height || c < 0 || c >=width){
		return;
	}
	
		int index = (r * width +c) * 3;

	if (newZval >= zbuffer[index]){
		zbuffer[index] = newZval;
		
		buffer[index + 0] = (unsigned char) ceil(col[0] * 255);
		buffer[index + 1] = (unsigned char) ceil(col[1] * 255);
		buffer[index + 2] = (unsigned char) ceil(col[2] * 255);
	}
}

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1e_geometry.vtk");
    cerr << "Reading" << endl;
    rdr->Update();
    cerr << "Done reading" << endl;
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();

    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkDoubleArray *var = (vtkDoubleArray *) pd->GetPointData()->GetArray("hardyglobal");
    double *color_ptr = var->GetPointer(0);
    //vtkFloatArray *var = (vtkFloatArray *) pd->GetPointData()->GetArray("hardyglobal");
    //float *color_ptr = var->GetPointer(0);
    vtkFloatArray *n = (vtkFloatArray *) pd->GetPointData()->GetNormals();
    float *normals = n->GetPointer(0);
    std::vector<Triangle> tris(numTris);
    vtkIdType npts;
    vtkIdType *ptIds;
    int idx;
    for (idx = 0, cells->InitTraversal() ; cells->GetNextCell(npts, ptIds) ; idx++)
    {
        if (npts != 3)
        {
            cerr << "Non-triangles!! ???" << endl;
            exit(EXIT_FAILURE);
        }
        double *pt = NULL;
        pt = pts->GetPoint(ptIds[0]);
        tris[idx].X[0] = pt[0];
        tris[idx].Y[0] = pt[1];
        tris[idx].Z[0] = pt[2];
#ifdef NORMALS
        tris[idx].normals[0][0] = normals[3*ptIds[0]+0];
        tris[idx].normals[0][1] = normals[3*ptIds[0]+1];
        tris[idx].normals[0][2] = normals[3*ptIds[0]+2];
#endif
        pt = pts->GetPoint(ptIds[1]);
        tris[idx].X[1] = pt[0];
        tris[idx].Y[1] = pt[1];
        tris[idx].Z[1] = pt[2];
#ifdef NORMALS
        tris[idx].normals[1][0] = normals[3*ptIds[1]+0];
        tris[idx].normals[1][1] = normals[3*ptIds[1]+1];
        tris[idx].normals[1][2] = normals[3*ptIds[1]+2];
#endif
        pt = pts->GetPoint(ptIds[2]);
        tris[idx].X[2] = pt[0];
        tris[idx].Y[2] = pt[1];
        tris[idx].Z[2] = pt[2];
#ifdef NORMALS
        tris[idx].normals[2][0] = normals[3*ptIds[2]+0];
        tris[idx].normals[2][1] = normals[3*ptIds[2]+1];
        tris[idx].normals[2][2] = normals[3*ptIds[2]+2];
#endif

        // 1->2 interpolate between light blue, dark blue
        // 2->2.5 interpolate between dark blue, cyan
        // 2.5->3 interpolate between cyan, green
        // 3->3.5 interpolate between green, yellow
        // 3.5->4 interpolate between yellow, orange
        // 4->5 interpolate between orange, brick
        // 5->6 interpolate between brick, salmon
        double mins[7] = { 1, 2, 2.5, 3, 3.5, 4, 5 };
        double maxs[7] = { 2, 2.5, 3, 3.5, 4, 5, 6 };
        unsigned char RGB[8][3] = { { 71, 71, 219 }, 
                                    { 0, 0, 91 },
                                    { 0, 255, 255 },
                                    { 0, 128, 0 },
                                    { 255, 255, 0 },
                                    { 255, 96, 0 },
                                    { 107, 0, 0 },
                                    { 224, 76, 76 } 
                                  };
        for (int j = 0 ; j < 3 ; j++)
        {
            float val = color_ptr[ptIds[j]];
            int r;
            for (r = 0 ; r < 7 ; r++)
            {
                if (mins[r] <= val && val < maxs[r])
                    break;
            }
            if (r == 7)
            {
                cerr << "Could not interpolate color for " << val << endl;
                exit(EXIT_FAILURE);
            }
            double proportion = (val-mins[r]) / (maxs[r]-mins[r]);
            tris[idx].colors[j][0] = (RGB[r][0]+proportion*(RGB[r+1][0]-RGB[r][0]))/255.0;
            tris[idx].colors[j][1] = (RGB[r][1]+proportion*(RGB[r+1][1]-RGB[r][1]))/255.0;
            tris[idx].colors[j][2] = (RGB[r][2]+proportion*(RGB[r+1][2]-RGB[r][2]))/255.0;
        }
    }

    return tris;
}



void getFrame(Camera cam, double O[3], double u[3], double v[3], double w[3]){
	//double O[3];
	//double u[3];
	//double v[3];
	//double w[3];
	double tempu[3];


	O[0] = cam.position[0];
	O[1] = cam.position[1];
	O[2] = cam.position[2];

	tempu[0] = O[0] - cam.focus[0];
	tempu[1] = O[1] - cam.focus[1];
	tempu[2] = O[2] - cam.focus[2];

	u[0] = cam.up[1] * tempu[2] - cam.up[2] * tempu[1];
	u[1] = tempu[0] * cam.up[2] - cam.up[0] * tempu[2];
	u[2] = cam.up[0] * tempu[1] - cam.up[1] * tempu[0];

	v[0] = tempu[1] * u[2] - tempu[2] * u[1];
	v[1] = u[0] * tempu[2] - tempu[0] * u[2];
	v[2] = tempu[0] * u[1] - tempu[1] * u[0];


	w[0] = tempu[0];
	w[1] = tempu[1];
	w[2] = tempu[2];


}




void normalizeValue(double A[3]){
	double B = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
	A[0] = A[0]/B;
	A[1] = A[1]/B;
	A[2] = A[2]/B;

}

double findDotProduct(double a[3], double t[3]){
	double product = ((a[0] * t[0]) + (a[1] * t[1]) + (a[2]*t[2]));
	return product;
}

Matrix CameraTransform(Camera c, double O[3], double u[3], double v[3], double w[3]){

	getFrame(c, O, u, v, w);
	normalizeValue(u);
	normalizeValue(v);
	normalizeValue(w);

	double t[3];
	t[0] = 0 - O[0];
	t[1] = 0 - O[1];
	t[2] = 0 - O[2];

	double dotu =findDotProduct(u,t);
	double dotv =findDotProduct(v,t);
	double dotw =findDotProduct(w,t);



	Matrix ct;

	ct.A[0][0] = u[0];
	ct.A[0][1] = v[0];
	ct.A[0][2] = w[0];
	ct.A[0][3] = 0;

	ct.A[1][0] = u[1];
	ct.A[1][1] = v[1];
	ct.A[1][2] = w[1];
	ct.A[1][3] = 0;

	ct.A[2][0] = u[2];
	ct.A[2][1] = v[2];
	ct.A[2][2] = w[2];
	ct.A[2][3] = 0;

	ct.A[3][0] = dotu;
	ct.A[3][1] = dotv;
	ct.A[3][2] = dotw;
	ct.A[3][3] = 1;

	return ct;
}


Matrix ViewTransform(double alpha, double n, double f){

	double cotval = (1/(tan(alpha/2)));

	Matrix vt;

	vt.A[0][0] = cotval;
	vt.A[0][1] = 0;
	vt.A[0][2] = 0;
	vt.A[0][3] = 0;

	vt.A[1][0] = 0;
	vt.A[1][1] = cotval;
	vt.A[1][2] = 0;
	vt.A[1][3] = 0;

	vt.A[2][0] = 0;
	vt.A[2][1] = 0;
	vt.A[2][2] = (f+n)/ (f-n);
	vt.A[2][3] = -1;

	vt.A[3][0] = 0;
	vt.A[3][1] = 0;
	vt.A[3][2] = (2*f*n)/(f-n);
	vt.A[3][3] = 0;

	return vt;

}



Matrix DeviceTransform(double n, double m){
	Matrix tt;

	tt.A[0][0] = n/2;
	tt.A[0][1] = 0;
	tt.A[0][2] = 0;
	tt.A[0][3] = 0;

	tt.A[1][0] = 0;
	tt.A[1][1] = m/2;
	tt.A[1][2] = 0;
	tt.A[1][3] = 0;

	tt.A[2][0] = 0;
	tt.A[2][1] = 0;
	tt.A[2][2] = 1;
	tt.A[2][3] = 0;

	tt.A[3][0] = n/2;
	tt.A[3][1] = m/2;
	tt.A[3][2] = 0;
	tt.A[3][3] = 1;

	return tt;
}

void CalculateShading(LightingParameters &lp, Triangle triangle, Camera c, double shading[3]){
	
	
	double R0[3];
	double R1[3];
	double R2[3];

	double viewDirection0[3];
	double viewDirection1[3];
	double viewDirection2[3];

	double dotln0 = findDotProduct(lp.lightDir, triangle.normals[0]);
	double dotln1 = findDotProduct(lp.lightDir, triangle.normals[1]);
	double dotln2 = findDotProduct(lp.lightDir, triangle.normals[2]);
	

	R0[0] = 2*(dotln0) * triangle.normals[0][0] - lp.lightDir[0];
	R0[1] = 2*(dotln0) * triangle.normals[0][1] - lp.lightDir[1];
	R0[2] = 2*(dotln0) * triangle.normals[0][2] - lp.lightDir[2];

	R1[0] = 2*(dotln1) * triangle.normals[1][0] - lp.lightDir[0];
	R1[1] = 2*(dotln1) * triangle.normals[1][1] - lp.lightDir[1];
	R1[2] = 2*(dotln1) * triangle.normals[1][2] - lp.lightDir[2];

	R2[0] = 2*(dotln2) * triangle.normals[2][0] - lp.lightDir[0];
	R2[1] = 2*(dotln2) * triangle.normals[2][1] - lp.lightDir[1];
	R2[2] = 2*(dotln2) * triangle.normals[2][2] - lp.lightDir[2];


	viewDirection0[0] = c.position[0] - triangle.X[0];
	viewDirection0[1] = c.position[1] - triangle.Y[0];
	viewDirection0[2] = c.position[2] - triangle.Z[0];

	viewDirection1[0] = c.position[0] - triangle.X[1];
	viewDirection1[1] = c.position[1] - triangle.Y[1];
	viewDirection1[2] = c.position[2] - triangle.Z[1];

	viewDirection2[0] = c.position[0] - triangle.X[2];
	viewDirection2[1] = c.position[1] - triangle.Y[2];
	viewDirection2[2] = c.position[2] - triangle.Z[2];

	//normalize values below

	normalizeValue(viewDirection0);
	normalizeValue(viewDirection1);
	normalizeValue(viewDirection2);

	normalizeValue(R0);
	normalizeValue(R1);
	normalizeValue(R2);


	double zerovalue = 0.0;


	//find shading[0]

	double dlight = abs(dotln0);

	double dotVR = findDotProduct(R0, viewDirection0);

	double cosa = dotVR;

	double slight = std::max(zerovalue, pow(cosa, lp.alpha));

	shading[0] = lp.Ka + lp.Kd * dlight + lp.Ks * slight;

	/*cout<< "Ambient: "<<lp.Ka<<endl;
	cout<<"dlight: "<<lp.Kd * dlight<<endl;
	cout<< "slight: "<<lp.Ks*slight<<endl;
	cout<<"shading[0]: "<<shading[0]<<endl;
*/

	//find shading[1]

	dlight = abs(dotln1);

	dotVR = findDotProduct(R1, viewDirection1);

	cosa = dotVR;

	slight = 0.0;
	slight = std::max(zerovalue, pow(cosa, lp.alpha));

	shading[1] = lp.Ka + lp.Kd * dlight + lp.Ks * slight;
/*
	cout<<"shading[1]: "<<shading[1]<<endl;
	cout<<"dlight1: "<<dlight<<endl;
	cout<<"slight1: "<<slight<<endl;

*/

	//find shading[2]

	dlight = abs(dotln2);

	dotVR = findDotProduct(R2, viewDirection2);

	cosa = dotVR;

	slight = 0.0;
	slight = std::max(zerovalue, pow(cosa, lp.alpha));

	shading[2] = lp.Ka + lp.Kd * dlight + lp.Ks * slight;

	/*cout<<"shading[2]: "<<shading[2]<<endl;
	cout<<"dlight2: "<<dlight<<endl;
	cout<<"slight2: "<<slight<<endl;
*/
}


void RasterizeGoingDownTriangle(Triangle triangles, int width, int height, unsigned char *buffer, Screen screen, double colors[3][3], double *zbuffer){
	int trTOb = 0;
	int lvTOb = 0;
	double a = 0.0;
	double B[3];
	double TR[3];
	double LV[3];
	double colB[3];
	double colTR[3];
	double colLV[3];
	double miny = 0.0;
	double my = 0.0;
	double mx = 0.0;
	double mtrb = 0.0;
	double mlvb = 0.0;
	double btrb = 0.0;
	double blvb = 0.0;
	double maxy = 0.0;
	double shadeB;
	double shadeTR;
	double shadeLV;
	int check = 0;


	a = std::min(triangles.Y[0], triangles.Y[1]);
	a = std::min(a,triangles.Y[2]);
	if(a == triangles.Y[0]){
		B[0] = triangles.X[0];
		B[1] = triangles.Y[0];
		B[2] = triangles.Z[0];
		colB[0] = triangles.colors[0][0];
		colB[1] = triangles.colors[0][1];
		colB[2] = triangles.colors[0][2];
		shadeB = triangles.shading[0];
		check = 0;
	}
	else if (a == triangles.Y[1]){
		B[0] = triangles.X[1];
		B[1] = triangles.Y[1];
		B[2] = triangles.Z[1];
		colB[0] = triangles.colors[1][0];
		colB[1] = triangles.colors[1][1];
		colB[2] = triangles.colors[1][2];
		shadeB = triangles.shading[1];
		check = 1;
	}
	else if( a == triangles.Y[2]){
		B[0] = triangles.X[2];
		B[1] = triangles.Y[2];
		B[2] = triangles.Z[2];
		colB[0] = triangles.colors[2][0];
		colB[1] = triangles.colors[2][1];
		colB[2] = triangles.colors[2][2];
		shadeB = triangles.shading[2];
		check = 2;
	}
	if (check == 0){
		if (triangles.X[1] > triangles.X[2]){
			TR[0] = triangles.X[1];
			TR[1] = triangles.Y[1];
			TR[2] = triangles.Z[1];
			colTR[0] = triangles.colors[1][0];
			colTR[1] = triangles.colors[1][1];
			colTR[2] = triangles.colors[1][2];
			shadeTR = triangles.shading[1];

			LV[0] = triangles.X[2];
			LV[1] = triangles.Y[2];
			LV[2] = triangles.Z[2];
			colLV[0] = triangles.colors[2][0];
			colLV[1] = triangles.colors[2][1];
			colLV[2] = triangles.colors[2][2];
			shadeLV = triangles.shading[2];
		}
		else{
			TR[0] = triangles.X[2];
			TR[1] = triangles.Y[2];
			TR[2] = triangles.Z[2];
			colTR[0] = triangles.colors[2][0];
			colTR[1] = triangles.colors[2][1];
			colTR[2] = triangles.colors[2][2];
			shadeTR = triangles.shading[2];

			LV[0] = triangles.X[1];
			LV[1] = triangles.Y[1];
			LV[2] = triangles.Z[1];
			colLV[0] = triangles.colors[1][0];
			colLV[1] = triangles.colors[1][1];
			colLV[2] = triangles.colors[1][2];
			shadeLV = triangles.shading[1];
		}
	}
	else if( check == 1){
		if(triangles.X[0] > triangles.X[2]){
			TR[0] = triangles.X[0];
			TR[1] = triangles.Y[0];
			TR[2] = triangles.Z[0];
			colTR[0] = triangles.colors[0][0];
			colTR[1] = triangles.colors[0][1];
			colTR[2] = triangles.colors[0][2];
			shadeTR = triangles.shading[0];


			LV[0] = triangles.X[2];
			LV[1] = triangles.Y[2];
			LV[2] = triangles.Z[2];
			colLV[0] = triangles.colors[2][0];
			colLV[1] = triangles.colors[2][1];
			colLV[2] = triangles.colors[2][2];
			shadeLV = triangles.shading[2];
		}
		else{
			TR[0] = triangles.X[2];
			TR[1] = triangles.Y[2];
			TR[2] = triangles.Z[2];
			colTR[0] = triangles.colors[2][0];
			colTR[1] = triangles.colors[2][1];
			colTR[2] = triangles.colors[2][2];
			shadeTR = triangles.shading[2];

			LV[0] = triangles.X[0];
			LV[1] = triangles.Y[0];
			LV[2] = triangles.Z[0];
			colLV[0] = triangles.colors[0][0];
			colLV[1] = triangles.colors[0][1];
			colLV[2] = triangles.colors[0][2];
			shadeLV = triangles.shading[0];
		}

	}
	else if (check == 2){
		if (triangles.X[0] > triangles.X[1]){
			TR[0] = triangles.X[0];
			TR[1] = triangles.Y[0];
			TR[2] = triangles.Z[0];
			colTR[0] = triangles.colors[0][0];
			colTR[1] = triangles.colors[0][1];
			colTR[2] = triangles.colors[0][2];
			shadeTR = triangles.shading[0];

			LV[0] = triangles.X[1];
			LV[1] = triangles.Y[1];
			LV[2] = triangles.Z[1];
			colLV[0] = triangles.colors[1][0];
			colLV[1] = triangles.colors[1][1];
			colLV[2] = triangles.colors[1][2];
			shadeLV = triangles.shading[1];
		}
		else{
			TR[0] = triangles.X[1];
			TR[1] = triangles.Y[1];
			TR[2] = triangles.Z[1];
			colTR[0] = triangles.colors[1][0];
			colTR[1] = triangles.colors[1][1];
			colTR[2] = triangles.colors[1][2];
			shadeTR = triangles.shading[1];

			LV[0] = triangles.X[0];
			LV[1] = triangles.Y[0];
			LV[2] = triangles.Z[0];
			colLV[0] = triangles.colors[0][0];
			colLV[1] = triangles.colors[0][1];
			colLV[2] = triangles.colors[0][2];
			shadeLV = triangles.shading[0];
		}
	}


//this is where I begin to use algebra
	my = TR[1] - B[1];
	mx = TR[0] - B[0];
	mtrb = my/mx;
	btrb = TR[1] - (mtrb*TR[0]);

	if(LV[0] != 0 || B[0] != 0){
	my = LV[1] - B[1];
	mx = LV[0] - B[0];
	mlvb =  my/mx;
	blvb =  LV[1] - (mlvb*LV[0]);
	}

	miny = B[1];
	maxy = std::max(TR[1], LV[1]);

	if (LV[0] == B[0]){
		lvTOb = 1;
	}
	if (TR[0] == B[0]){
		trTOb = 1;
	}

	//cout<<"DOWN TRIANGLE PRE LOOP"<<endl;
	//follows pseudocode from class slides
	for(double r = ceil(miny); r<= floor(maxy); r++){
		double leftEnd = 0.0;
		double rightEnd = 0.0;
		
		leftEnd = (r - blvb)/mlvb;
		rightEnd = (r - btrb)/mtrb;
		if( lvTOb ==1){
			leftEnd = B[0];
		}
		if(trTOb == 1){
			rightEnd = B[0];
		}
		

		//testing out switching LV and B
		double tvalLeft = (leftEnd - B[0])/(LV[0] - B[0]);
		double tvalRight = (rightEnd - B[0])/(TR[0]-B[0]);

		//cout<<"LeftEnd: "<<leftEnd<<" B[0]: "<<B[0]<<" LV[0]: "<<LV[0]<<endl;

		if (B[0] == LV[0]) tvalLeft = (r - miny)/(maxy - miny);
		if (B[0] == TR[0]) tvalRight = (r - miny)/(maxy - miny);

		double LeftZ = B[2] + tvalLeft*(LV[2] - B[2]);
		double RightZ = B[2] + tvalRight*(TR[2] - B[2]);

		double LeftR = colB[0] + tvalLeft*(colLV[0] - colB[0]);
		double RightR = colB[0] + tvalRight*(colTR[0] - colB[0]);

		double LeftG = colB[1] + tvalLeft*(colLV[1] - colB[1]);
		double RightG = colB[1] + tvalRight*(colTR[1] - colB[1]);

		double LeftB = colB[2] + tvalLeft*(colLV[2] - colB[2]);
		double RightB = colB[2] + tvalRight*(colTR[2] - colB[2]);

		double LeftShade = shadeB + tvalLeft*(shadeLV - shadeB);
		double RightShade = shadeB + tvalLeft*(shadeTR - shadeB);


		//cerr<<""<<endl;
		//cerr<<"Rasterizing along row "<<r<<" with left end  = "<< leftEnd<<" (Z: "<<LeftZ<<", RGB = "<<LeftR<<"/"<<LeftG<<"/"<<LeftB<<") and right end " << rightEnd << "(Z: "<< RightZ<<", RGB = "<<RightR<<"/"<<RightG<<"/"<<RightB<<")"<<endl;
		//cerr<<""<<endl;

		//cout<<"DOWN TRIANGLE LOOP 1 END"<<endl;

//if(rightEnd - leftEnd > 20) abort();
		for(double c = ceil(leftEnd); c <= floor(rightEnd); c++){
			//use SetPixel to draw the triangles on the screen

			double newtval = (c - leftEnd)/(rightEnd - leftEnd);

			//interpolate between left and right ends
			double newZval = LeftZ + newtval*(RightZ - LeftZ);

			double newRval = LeftR + newtval*(RightR - LeftR);
			double newGval = LeftG + newtval*(RightG - LeftG);
			double newBval = LeftB + newtval*(RightB - LeftB);

			double newShade = LeftShade + newtval*(RightShade - LeftShade);

			double colorAtPoint[3];
			colorAtPoint[0] = std::min(1.0, newRval * newShade);
			colorAtPoint[1] = std::min(1.0, newGval * newShade);
			colorAtPoint[2] = std::min(1.0, newBval * newShade);
			
			//cerr<<"Got fragment r = "<< r<< ", c = "<< c<<" z = "<<newZval<<", color = "<<newRval<<"/"<<newGval<<"/"<<newBval<<endl;


			screen.SetPixel(r, c, colorAtPoint,zbuffer, newZval, newShade); 
			//cout<<"DOWN TRIANGLE LOOP 2 END"<<endl;
		}
	}




}





void RasterizeGoingUpTriangle(Triangle triangles, int width, int height, unsigned char *buffer, Screen screen, double colors[3][3], double *zbuffer){
	int trTOb = 0;
	int lvTOb = 0;
	double a = 0.0;
	double B[3];
	double TR[3];
	double LV[3];
	double colB[3];
	double colTR[3];
	double colLV[3];
	double miny = 0.0;
	double my = 0.0;
	double mx = 0.0;
	double mtrb = 0.0;
	double mlvb = 0.0;
	double btrb = 0.0;
	double blvb = 0.0;
	double maxy = 0.0;
	double shadeB;
	double shadeTR;
	double shadeLV;
	int check = 0;
	

	a = std::max(triangles.Y[0], triangles.Y[1]);
	a = std::max(a,triangles.Y[2]);
	if(a== triangles.Y[0]){
		B[0] = triangles.X[0];
		B[1] = triangles.Y[0];
		B[2] = triangles.Z[0];
		colB[0] = triangles.colors[0][0];
		colB[1] = triangles.colors[0][1];
		colB[2] = triangles.colors[0][2];
		shadeB = triangles.shading[0];
		check = 0;
	}
	else if (a == triangles.Y[1]){
		B[0] = triangles.X[1];
		B[1] = triangles.Y[1];
		B[2] = triangles.Z[1];
		colB[0] = triangles.colors[1][0];
		colB[1] = triangles.colors[1][1];
		colB[2] = triangles.colors[1][2];
		shadeB = triangles.shading[1];
		check = 1;
	}
	else if( a == triangles.Y[2]){
		B[0] = triangles.X[2];
		B[1] = triangles.Y[2];
		B[2] = triangles.Z[2];
		colB[0] = triangles.colors[2][0];
		colB[1] = triangles.colors[2][1];
		colB[2] = triangles.colors[2][2];
		shadeB = triangles.shading[2];
		check = 2;
	}
	if (check == 0){
		if (triangles.X[1] > triangles.X[2]){
			TR[0] = triangles.X[1];
			TR[1] = triangles.Y[1];
			TR[2] = triangles.Z[1];
			colTR[0] = triangles.colors[1][0];
			colTR[1] = triangles.colors[1][1];
			colTR[2] = triangles.colors[1][2];
			shadeTR = triangles.shading[1];
			

			LV[0] = triangles.X[2];
			LV[1] = triangles.Y[2];
			LV[2] = triangles.Z[2];
			colLV[0] = triangles.colors[2][0];
			colLV[1] = triangles.colors[2][1];
			colLV[2] = triangles.colors[2][2];
			shadeLV = triangles.shading[2];
		}
		else{
			TR[0] = triangles.X[2];
			TR[1] = triangles.Y[2];
			TR[2] = triangles.Z[2];
			colTR[0] = triangles.colors[2][0];
			colTR[1] = triangles.colors[2][1];
			colTR[2] = triangles.colors[2][2];
			shadeTR = triangles.shading[2];


			LV[0] = triangles.X[1];
			LV[1] = triangles.Y[1];
			LV[2] = triangles.Z[1];
			colLV[0] = triangles.colors[1][0];
			colLV[1] = triangles.colors[1][1];
			colLV[2] = triangles.colors[1][2];
			shadeLV = triangles.shading[1];
		}
	}
	else if( check == 1){
		if(triangles.X[0] > triangles.X[2]){
			TR[0] = triangles.X[0];
			TR[1] = triangles.Y[0];
			TR[2] = triangles.Z[0];
			colTR[0] = triangles.colors[0][0];
			colTR[1] = triangles.colors[0][1];
			colTR[2] = triangles.colors[0][2];
			shadeTR = triangles.shading[0];

			LV[0] = triangles.X[2];
			LV[1] = triangles.Y[2];
			LV[2] = triangles.Z[2];
			colLV[0] = triangles.colors[2][0];
			colLV[1] = triangles.colors[2][1];
			colLV[2] = triangles.colors[2][2];
			shadeLV = triangles.shading[2];
		}
		else{
			TR[0] = triangles.X[2];
			TR[1] = triangles.Y[2];
			TR[2] = triangles.Z[2];
			colTR[0] = triangles.colors[2][0];
			colTR[1] = triangles.colors[2][1];
			colTR[2] = triangles.colors[2][2];
			shadeTR = triangles.shading[2];

			LV[0] = triangles.X[0];
			LV[1] = triangles.Y[0];
			LV[2] = triangles.Z[0];
			colLV[0] = triangles.colors[0][0];
			colLV[1] = triangles.colors[0][1];
			colLV[2] = triangles.colors[0][2];
			shadeLV = triangles.shading[0];
		}

	}
	else if (check == 2){
		if (triangles.X[0] > triangles.X[1]){
			TR[0] = triangles.X[0];
			TR[1] = triangles.Y[0];
			TR[2] = triangles.Z[0];
			colTR[0] = triangles.colors[0][0];
			colTR[1] = triangles.colors[0][1];
			colTR[2] = triangles.colors[0][2];
			shadeTR = triangles.shading[0];

			LV[0] = triangles.X[1];
			LV[1] = triangles.Y[1];
			LV[2] = triangles.Z[1];
			colLV[0] = triangles.colors[1][0];
			colLV[1] = triangles.colors[1][1];
			colLV[2] = triangles.colors[1][2];
			shadeLV = triangles.shading[1];
		}
		else{
			TR[0] = triangles.X[1];
			TR[1] = triangles.Y[1];
			TR[2] = triangles.Z[1];
			colTR[0] = triangles.colors[1][0];
			colTR[1] = triangles.colors[1][1];
			colTR[2] = triangles.colors[1][2];
			shadeTR = triangles.shading[1];

			LV[0] = triangles.X[0];
			LV[1] = triangles.Y[0];
			LV[2] = triangles.Z[0];
			colLV[0] = triangles.colors[0][0];
			colLV[1] = triangles.colors[0][1];
			colLV[2] = triangles.colors[0][2];
			shadeLV = triangles.shading[0];

		}
	}


	
/*
printf("TR[0]: %f\n",TR[0]);
printf("TR[0]: %f\n",TR[1]);
printf("LV[0]: %f\n",LV[0]);
printf("LV[1]: %f\n",LV[1]);
printf("B[0]: %f\n",B[0]);
printf("B[1]: %f\n",B[1]);
printf("\n");
*/
//this is where I begin to use algebra

	my = TR[1] - B[1];
	mx = TR[0] - B[0];
	mtrb = my/mx;
	btrb = TR[1] - (mtrb*TR[0]);

	if(LV[0] != 0 || B[0] != 0){
	my = LV[1] - B[1];
	mx = LV[0] - B[0];
	mlvb =  my/mx;
	blvb =  LV[1] - (mlvb*LV[0]);
	}

	maxy = B[1];
	miny = std::min(TR[1], LV[1]);

	if (LV[0] == B[0]){
		lvTOb = 1;
	}
	if (TR[0] == B[0]){
		trTOb = 1;
	}
	//cout<<"UP TRIANGLE PRE LOOP"<<endl;
	//follows pseudocode from class slides
	for(double r = ceil(miny); r<= floor(maxy); r++){
		double leftEnd = 0.0;
		double rightEnd = 0.0;
		leftEnd = (r - blvb)/mlvb;
		rightEnd = (r - btrb)/mtrb;
		if( lvTOb ==1){
			leftEnd = B[0];
		}
		if(trTOb == 1){
			rightEnd = B[0];
		}

		double tvalLeft = (leftEnd - LV[0])/(B[0] - LV[0]);
		double tvalRight = (rightEnd - TR[0])/(B[0]-TR[0]);
		if (B[0] == LV[0]) tvalLeft = (r - miny)/(maxy - miny);
		if (B[0] == TR[0]) tvalRight = (r - miny)/(maxy - miny);
		//cerr<<"tvalLeft: "<<tvalLeft<< " tvalRight: "<<tvalRight<<endl;
		//do I need to use Y values here? ^

		double LeftZ = LV[2] + tvalLeft*(B[2] - LV[2]);
		double RightZ = TR[2] + tvalRight*(B[2] - TR[2]);

		double LeftR = colLV[0] + tvalLeft*(colB[0] - colLV[0]);
		double RightR = colTR[0] + tvalRight*(colB[0] - colTR[0]);

		double LeftG = colLV[1] + tvalLeft*(colB[1] - colLV[1]);
		double RightG = colTR[1] + tvalRight*(colB[1] - colTR[1]);

		double LeftB = colLV[2] + tvalLeft*(colB[2] - colLV[2]);
		double RightB = colTR[2] + tvalRight*(colB[2] - colTR[2]);

		double LeftShade = shadeLV + tvalLeft*(shadeB - shadeLV);
		double RightShade = shadeTR + tvalRight*(shadeB - shadeTR);

		//cerr<<""<<endl;
		//cerr<<"Rasterizing along row "<<r<<" with left end  = "<< leftEnd<<" (Z: "<<LeftZ<<", RGB = "<<LeftR<<"/"<<LeftG<<"/"<<LeftB<<") and right end " << rightEnd << "(Z: "<< RightZ<<", RGB = "<<RightR<<"/"<<RightG<<"/"<<RightB<<")"<<endl;
		//cerr<<""<<endl;
		//interpolate to find z and rgb of left and right end

//if(rightEnd - leftEnd > 20) abort();
		//cout<<"UP TRIANGLE END LOOP 1"<<endl;
		for(double c = ceil(leftEnd); c <= floor(rightEnd); c++){

			double newtval = (c - leftEnd)/(rightEnd - leftEnd);


			//interpolate between left and right ends
			double newZval = LeftZ + newtval*(RightZ - LeftZ);
			

			double newRval = LeftR + newtval*(RightR - LeftR);
			double newGval = LeftG + newtval*(RightG - LeftG);
			double newBval = LeftB + newtval*(RightB - LeftB);

			double newShade = LeftShade + newtval*(RightShade - LeftShade);

			double colorAtPoint[3];
			colorAtPoint[0] = std::min(1.0, newRval * newShade);
			colorAtPoint[1] = std::min(1.0, newGval * newShade);
			colorAtPoint[2] = std::min(1.0, newBval * newShade);

			//cerr<<"Got fragment r = "<< r<< ", c = "<< c<<" z = "<<newZval<<", color = "<<newRval<<"/"<<newGval<<"/"<<newBval<<endl;
			//cerr<< "TVAL IS: "<<newtval<<endl;
			//use SetPixel to draw the triangles on the screen
			screen.SetPixel(r, c, colorAtPoint, zbuffer, newZval, newShade); 
			//cout<<"UP TRIANGLE END LOOP 2"<<endl;
		}
	}




}





void RasterizeArbitraryTriangle(Triangle triangles, int width, int height, unsigned char *buffer, Screen screen, double colors[3][3], double *zbuffer){
    double a;
    double B[2];
    double LV[2];
    double miny = 0.0;
    double my = 0.0;
    double mx = 0.0;
    double mtrb = 0.0;
    double mlvb = 0.0;
    double btrb = 0.0;
    double blvb = 0.0;
    double maxy = 0.0;
    double TR[2];
    double bot[3];
    double top[3];
    double mid[3];
    double colbot[3];
    double coltop[3];
    double colmid[3];
    double shadingbot;
    double shadingtop;
    double shadingmid;

    //find bottom point with min y
    miny = std::min(triangles.Y[0], triangles.Y[1]);
                miny = std::min(triangles.Y[2], miny);
                if (miny == triangles.Y[0]){
                        bot[0] = triangles.X[0];
                        bot[1] = triangles.Y[0];
                        bot[2] = triangles.Z[0];
                        colbot[0] = triangles.colors[0][0];
                        colbot[1] = triangles.colors[0][1];
                        colbot[2] = triangles.colors[0][2];
                        shadingbot = triangles.shading[0];




                }
                else if(miny == triangles.Y[1]){
                        bot[0] = triangles.X[1];
                        bot[1] = triangles.Y[1];
                        bot[2] = triangles.Z[1];
                        colbot[0] = triangles.colors[1][0];
                        colbot[1] = triangles.colors[1][1];
                        colbot[2] = triangles.colors[1][2];
                        shadingbot = triangles.shading[1];

                }
                else {
                        bot[0] = triangles.X[2];
                        bot[1] = triangles.Y[2];
                        bot[2] = triangles.Z[2];
                        colbot[0] = triangles.colors[2][0];
                        colbot[1] = triangles.colors[2][1];
                        colbot[2] = triangles.colors[2][2];
                        shadingbot = triangles.shading[2];
                }
            //find upper point with max y
            maxy = std::max(triangles.Y[0], triangles.Y[1]);
                maxy = std::max(triangles.Y[2], maxy);
                if (maxy == triangles.Y[0]){
                        top[0] = triangles.X[0];
                        top[1] = triangles.Y[0];
                        top[2] = triangles.Z[0];
                        coltop[0] = triangles.colors[0][0];
                        coltop[1] = triangles.colors[0][1];
                        coltop[2] = triangles.colors[0][2];
                        shadingtop = triangles.shading[0];

                }
                else if(maxy == triangles.Y[1]){
                        top[0] = triangles.X[1];
                        top[1] = triangles.Y[1];
                        top[2] = triangles.Z[1];
                        coltop[0] = triangles.colors[1][0];
                        coltop[1] = triangles.colors[1][1];
                        coltop[2] = triangles.colors[1][2];
                        shadingtop = triangles.shading[1];
                }
                else {
                        top[0] = triangles.X[2];
                        top[1] = triangles.Y[2];
                        top[2] = triangles.Z[2];
                        coltop[0] = triangles.colors[2][0];
                        coltop[1] = triangles.colors[2][1];
                        coltop[2] = triangles.colors[2][2];
                        shadingtop = triangles.shading[2];
                }
            //find remaining vertex
            //changed top[0] to be top[1] and second triangles to be triangles.Y instead of triangles.X
            if (triangles.Y[0] != bot[1] && triangles.Y[0] != top[1]){
                mid[0] = triangles.X[0];
                mid[1] = triangles.Y[0];
                mid[2] = triangles.Z[0];
                colmid[0] = triangles.colors[0][0];
                colmid[1] = triangles.colors[0][1];
                colmid[2] = triangles.colors[0][2];
                shadingmid = triangles.shading[0];

            }
            else if (triangles.Y[1] != bot[1] && triangles.Y[1] != top[1]){
                mid[0] = triangles.X[1];
                mid[1] = triangles.Y[1];
                mid[2] = triangles.Z[1];
                colmid[0] = triangles.colors[1][0];
                colmid[1] = triangles.colors[1][1];
                colmid[2] = triangles.colors[1][2];
                shadingmid = triangles.shading[1];
            }   
            else if (triangles.Y[2] != bot[1] && triangles.Y[2] != top[1]){
                mid[0] = triangles.X[2];
                mid[1] = triangles.Y[2];
                mid[2] = triangles.Z[2];
                colmid[0] = triangles.colors[2][0];
                colmid[1] = triangles.colors[2][1];
                colmid[2] = triangles.colors[2][2];
                shadingmid = triangles.shading[2];
            }
            //if slope is infinity, set x == top[0]
            //find where to cut triangle in half



            double slopetopbot = 0.0;
            double btopbot = 0.0;
            double xval;
            my = top[1] - bot[1];
            mx = top[0] - bot[0];
            slopetopbot = my/mx;
            double tup = (mid[1] - bot[1])/(top[1] - bot[1]);
            if (top[0] == bot[0]) {
                xval = top[0];
            }
            else{
            btopbot = bot[1] - (slopetopbot * bot[0]);
            //find midpoint x value
            //changed top[1] to bot[1] below
            //xval = (mid[1] - btopbot)/slopetopbot;
            xval = (bot[0] + tup*(top[0] - bot[0]));
            }


            //double tup = (mid[1] - bot[1])/(top[1] - bot[1]);
            double zval = (bot[2] + tup*(top[2] - bot[2]));
            

            double newcolorupR = colbot[0] + tup*(coltop[0] - colbot[0]);
            double newcolorupG = colbot[1] + tup*(coltop[1] - colbot[1]);
            double newcolorupB = colbot[2] + tup*(coltop[2] - colbot[2]);
            
            double newshading = shadingbot + tup*(shadingtop - shadingbot);


            //make new up triangle
            Triangle newupTriangle;

            newupTriangle.X[0] = xval;
            newupTriangle.Y[0] = mid[1];
            newupTriangle.Z[0] = zval;

            newupTriangle.X[1] = mid[0];
            newupTriangle.Y[1] = mid[1];
            newupTriangle.Z[1] = mid[2];

            newupTriangle.X[2] = top[0];
            newupTriangle.Y[2] = top[1];
            newupTriangle.Z[2] = top[2];

            newupTriangle.colors[0][0] = newcolorupR;
            newupTriangle.colors[0][1] = newcolorupG;
            newupTriangle.colors[0][2] = newcolorupB;
            newupTriangle.shading[0] = newshading;

            newupTriangle.colors[1][0] = colmid[0];
            newupTriangle.colors[1][1] = colmid[1];
            newupTriangle.colors[1][2] = colmid[2];
            newupTriangle.shading[1] = shadingmid;

            newupTriangle.colors[2][0] = coltop[0];
            newupTriangle.colors[2][1] = coltop[1];
            newupTriangle.colors[2][2] = coltop[2];
            newupTriangle.shading[2] = shadingtop;

            //make new down triangle
            Triangle newdownTriangle;

            newdownTriangle.X[0] = xval;
            newdownTriangle.Y[0] = mid[1];
            newdownTriangle.Z[0] = zval;

            newdownTriangle.X[1] = mid[0];
            newdownTriangle.Y[1] = mid[1];
            newdownTriangle.Z[1] = mid[2];

            newdownTriangle.X[2] = bot[0];
            newdownTriangle.Y[2] = bot[1];
            newdownTriangle.Z[2] = bot[2];

            newdownTriangle.colors[0][0] = newcolorupR;
            newdownTriangle.colors[0][1] = newcolorupG;
            newdownTriangle.colors[0][2] = newcolorupB;
            newdownTriangle.shading[0] = newshading;

            newdownTriangle.colors[1][0] = colmid[0];
            newdownTriangle.colors[1][1] = colmid[1];
            newdownTriangle.colors[1][2] = colmid[2];
            newdownTriangle.shading[1] = shadingmid;

            newdownTriangle.colors[2][0] = colbot[0];
            newdownTriangle.colors[2][1] = colbot[1];
            newdownTriangle.colors[2][2] = colbot[2];
            newdownTriangle.shading[2] = shadingbot;



        RasterizeGoingDownTriangle(newdownTriangle, screen.width, screen.height, screen.buffer, screen, newdownTriangle.colors, screen.zbuffer);
        RasterizeGoingUpTriangle(newupTriangle, screen.width, screen.height, screen.buffer, screen, newupTriangle.colors, screen.zbuffer);

}






int main()
{
	
   vtkImageData *image = NewImage(1000, 1000);
   unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
   int npixels = 1000*1000;

   double *zbuffer = new double [npixels * 3];
   for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;

   std::vector<Triangle> t = GetTriangles();

    Screen screen;
    screen.buffer = buffer;
    screen.width = 1000;
    screen.height = 1000;
    screen.zbuffer = zbuffer;



   for(int k = 0; k < npixels * 3; k++){
   		screen.zbuffer[k] = -1;
   }

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM

// variables to be used later
	double a;
	double B[2];
	double LV[2];
	double miny = 0.0;
	double my = 0.0;
	double mx = 0.0;
	double mtrb = 0.0;
	double mlvb = 0.0;
	double btrb = 0.0;
	double blvb = 0.0;
	double maxy = 0.0;
	double TR[2];
	double bot[3];
	double top[3];
	double mid[3];
	double colbot[3];
	double coltop[3];
	double colmid[3];
	double O[3];
	double u[3];
	double v[3];
	double w[3];
	int d = 0;
//beginning for loop

//currently set to only create one image (frame000)
//in order to create all 1000 images, simply set the for loop to go up to 1000
//instead of to 1
	for(int im = 0; im < 1; im++){

	Screen screen;
    screen.buffer = buffer;
    screen.width = 1000;
    screen.height = 1000;
    screen.zbuffer = zbuffer;

    for(int k = 0; k < npixels * 3; k++){
   	screen.zbuffer[k] = -1;
    }

    for (int i = 0 ; i < npixels*3 ; i++){
       screen.buffer[i] = 0;
    }


	Camera c1 = GetCamera(im,1000);
	
	
	Matrix ct = CameraTransform(c1, O, u, v, w);

	/*cout<<"Camera Transform: "<<endl;
	for(int i = 0; i<4; i++){
		for(int j = 0; j<4; j++){
			cout<<"Position: "<<i<<", "<<j<<endl;
			cout<<ct.A[i][j]<<endl;
		}
	}
*/
	Matrix vt = ViewTransform(c1.angle, c1.near, c1.far);
	/*
	cout<<"View Transform: "<<endl;
		for(int i = 0; i<4; i++){
			for(int j = 0; j<4; j++){
			cout<<"Position: "<<i<<", "<<j<<endl;
			cout<<vt.A[i][j]<<endl;
		}
	}
*/
	//Matix dt = DeviceTransform()

	Matrix ctvt = Matrix::ComposeMatrices(ct, vt);

	Matrix dt = DeviceTransform(screen.width,screen.height);

	Matrix tm = Matrix::ComposeMatrices(ctvt, dt);
		
/*
		cout<<"Total Transform: "<<endl;
		for(int i = 0; i<4; i++){
			for(int j = 0; j<4; j++){
			cout<<"Position: "<<i<<", "<<j<<endl;
			cout<<tm.A[i][j]<<endl;
		}
	}
*/

//this will find the top right vertex and put it to the TR list
	for( int j = 0; j < t.size(); j++){




		double transformpointsin[4];
		double zeropoint[4];
		double onepoint[4];
		double twopoint[4];

		transformpointsin[0] = t[j].X[0];
		transformpointsin[1] = t[j].Y[0];
		transformpointsin[2] = t[j].Z[0];
		transformpointsin[3] = 1;

		tm.TransformPoint(transformpointsin, zeropoint);

		transformpointsin[0] = t[j].X[1];
		transformpointsin[1] = t[j].Y[1];
		transformpointsin[2] = t[j].Z[1];
		transformpointsin[3] = 1;

		tm.TransformPoint(transformpointsin, onepoint);

		transformpointsin[0] = t[j].X[2];
		transformpointsin[1] = t[j].Y[2];
		transformpointsin[2] = t[j].Z[2];
		transformpointsin[3] = 1;

		tm.TransformPoint(transformpointsin, twopoint);


		Triangle triangles;

		triangles.X[0] = zeropoint[0] / zeropoint[3];
		triangles.X[1] = onepoint[0] / onepoint[3];
		triangles.X[2] = twopoint[0] / twopoint[3];

		triangles.Y[0] = zeropoint[1] / zeropoint[3];
		triangles.Y[1] = onepoint[1] / onepoint[3];
		triangles.Y[2] = twopoint[1] / twopoint[3];

		triangles.Z[0] = zeropoint[2] / zeropoint[3];
		triangles.Z[1] = onepoint[2] / onepoint[3];
		triangles.Z[2] = twopoint[2] / twopoint[3];

		triangles.colors[0][0] = t[j].colors[0][0];
		triangles.colors[0][1] = t[j].colors[0][1];
		triangles.colors[0][2] = t[j].colors[0][2];

		triangles.colors[1][0] = t[j].colors[1][0];
		triangles.colors[1][1] = t[j].colors[1][1];
		triangles.colors[1][2] = t[j].colors[1][2];

		triangles.colors[2][0] = t[j].colors[2][0];
		triangles.colors[2][1] = t[j].colors[2][1];
		triangles.colors[2][2] = t[j].colors[2][2];

		triangles.normals[0][0] = t[j].normals[0][0];
		triangles.normals[0][1] = t[j].normals[0][1];
		triangles.normals[0][2] = t[j].normals[0][2];

		triangles.normals[1][0] = t[j].normals[1][0];
		triangles.normals[1][1] = t[j].normals[1][1];
		triangles.normals[1][2] = t[j].normals[1][2];

		triangles.normals[2][0] = t[j].normals[2][0];
		triangles.normals[2][1] = t[j].normals[2][1];
		triangles.normals[2][2] = t[j].normals[2][2];




		double shade[3];

		CalculateShading(lp, t[j], c1, shade);

		triangles.shading[0] = shade[0];
		triangles.shading[1] = shade[1];
		triangles.shading[2] = shade[2];



		int trTOb = 0;
		int lvTOb = 0;
		
		if(triangles.Y[0] == triangles.Y[1] || triangles.Y[0] == triangles.Y[2] || triangles.Y[1] == triangles.Y[2] ){
			double tempmin = std::min(triangles.Y[0], triangles.Y[1]);
			tempmin = std::min(tempmin, triangles.Y[2]);
			int checker = 0;
			if (tempmin == triangles.Y[0]){
				if (triangles.Y[0] == triangles.Y[1] || triangles.Y[2] == triangles.Y[0]){
					checker = 1;	
				}	
			}
			else if (tempmin == triangles.Y[1]){
				if (triangles.Y[1] == triangles.Y[0] || triangles.Y[2] == triangles.Y[1]){
					checker = 1;
				}
			}
			else{
				if (triangles.Y[2] == triangles.Y[1] || triangles.Y[0] == triangles.Y[2]){
					checker = 1;
				}
			}
			if (checker == 0){

				RasterizeGoingDownTriangle(triangles, screen.width, screen.height, screen.buffer, screen, triangles.colors, screen.zbuffer);
			}
			else if(checker ==1){

				RasterizeGoingUpTriangle(triangles, screen.width, screen.height, screen.buffer, screen, triangles.colors, screen.zbuffer);
			}
		}

		else{
			
          RasterizeArbitraryTriangle(triangles, screen.width, screen.height, screen.buffer, screen, triangles.colors, screen.zbuffer);

}


}


WriteImage(image, "test");


  //deleting the zbuffer
  
}
delete[] zbuffer;
}
