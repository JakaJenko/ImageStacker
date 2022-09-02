__kernel void kernelCombineImages(
    int width,
    int height,
    int moveWidth,
    int moveHeight,
    __global uchar* mainImage,
    __global uchar* imageToAdd,
    int sampleCount)
{
    
    int index = get_global_id(0);

    if (index > height * width)
    {
        return;
    }

    int w = (index % (width));
    int h = (int)(floor((float)(index / (width))));

    int wMoved = (w + moveWidth) % width;
    int hMoved = (h + moveHeight) % height;

    int orgPos = (h * width + w) * 4;
    int movedPos = (hMoved * width + wMoved) * 4;

    
    mainImage[orgPos] = mainImage[orgPos] + (imageToAdd[movedPos] - mainImage[orgPos]) / sampleCount;
    mainImage[orgPos + 1] = mainImage[orgPos + 1] + (imageToAdd[movedPos + 1] - mainImage[orgPos + 1]) / sampleCount;
    mainImage[orgPos + 2] = mainImage[orgPos + 2] + (imageToAdd[movedPos + 2] - mainImage[orgPos + 2]) / sampleCount;
    mainImage[orgPos + 3] = mainImage[orgPos + 3] + (imageToAdd[movedPos + 3] - mainImage[orgPos + 3]) / sampleCount;
    
    /*
    if (w == 1640 && h == 1100)
    {
        printf("%d %d %d", index, mainImage[orgPos], imageToAdd[orgPos]);
    }*/
    /*
    mainImage[orgPos] = (mainImage[orgPos] + imageToAdd[orgPos]) / 2;
    mainImage[orgPos + 1] = (mainImage[orgPos + 1] + imageToAdd[orgPos + 1]) / 2;
    mainImage[orgPos + 2] = (mainImage[orgPos + 2] + imageToAdd[orgPos + 2]) / 2;
    mainImage[orgPos + 3] = (mainImage[orgPos + 3] + imageToAdd[orgPos + 3]) / 2;
    */

    /*
    2 6 4 9 4 = 25 / 5 = 5
    2 6 4 9 4 = 
        0 + (2 - 0) / 1 = 0 + 2 / 1 = 0 + 2 = 2
        2 + (6 - 2) / 2 = 2 + 4 / 2 = 2 + 2 = 4
        4 + (4 - 4) / 3 = 4
        4 + (9 - 4) / 4 = 4 + 5 / 4 = 4 + 1.25 = 5.25
        5.25 + (4 - 5.25) / 5 = 5.25 - 1.25 / 5 = 5.25 - 0.25 = 5
     */
}