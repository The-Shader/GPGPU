class GpuManager {
	
	public:

		GpuManager();

        void clearBuffers();

        void initBuffers();

        void setSize(const unsigned int inputSize);

        void copyDataImageToDevice(float *redBuffer, float *greenBuffer, float *blueBuffer);

        void generateAndCopyRandomSeedsToDevice();

        void copyDataFromDevice(float *redBuffer, float *greenBuffer, float *blueBuffer);

        void copyImageDataFromDevice(float *palette, unsigned short *indexes);

        void quantizeImage();

        int getNumOfClusters();


	private:
        
		float *dRedBuffer;

    	float *dGreenBuffer;

    	float *dBlueBuffer;

    	float *centroids;

    	float *newCentroids;

    	int *centroidsQuantity;

    	float *distanceAccumulation;

    	unsigned short *labels;

    	unsigned int *dSeeds;

    	unsigned int size;

    	int numOfClusters;
};
