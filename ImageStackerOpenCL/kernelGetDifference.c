__kernel void kernelGetDifference(int width,
    int height,
    int moveWidth,
    int moveHeight,
    __global uchar* originalImg,
    __global uchar* moveImage,
    __global int* partialSums,
    __local int* localSums)
{
    
    int index = get_global_id(0);
    
    if (index > height * width)
    {
        return;
    }
    
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);


    int w = (index % (width));
    int h = (int)(floor((float)(index / (width))));

    int wMoved = (w + moveWidth) % width;
    int hMoved = (h + moveHeight) % height;

    //int imgH = height;// / 4;

    int orgPos = (h * width + w) * 4;
    int movedPos = (hMoved * width + wMoved) * 4;
    
    /*
    if (w == 1640 && h == 1100)
    //if (w == 500 && h == 500)
    {
        printf("%d %d %d %d\n", index, w, h,  orgPos);
        printf("%d %d %d\n", index, originalImg[orgPos], moveImage[orgPos]);
    }*/

    /*
    //if (w == 1346 && h == 2460)
    if(abs_diff(moveImage[movedPos], originalImg[orgPos]) + abs_diff(moveImage[movedPos + 1], originalImg[orgPos + 1]) + abs_diff(moveImage[movedPos + 2], originalImg[orgPos + 2]) != 0)
    {
        printf("%d %d | %d %d\n", w, h, wMoved, hMoved);
        printf("%d %d %d\n", index, orgPos, movedPos);
        printf("%d\n\n", abs_diff(moveImage[movedPos], originalImg[orgPos]) + abs_diff(moveImage[movedPos + 1], originalImg[orgPos + 1]) + abs_diff(moveImage[movedPos + 2], originalImg[orgPos + 2]));
    }
    */
    

    localSums[local_id] = abs_diff(moveImage[movedPos], originalImg[orgPos]) + abs_diff(moveImage[movedPos + 1], originalImg[orgPos + 1]) + abs_diff(moveImage[movedPos + 2], originalImg[orgPos + 2]);  //R G B
    //localSums[local_id]
    
    
    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride = group_size / 2; stride > 0; stride /= 2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }

    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
        partialSums[get_group_id(0)] = localSums[0];
}