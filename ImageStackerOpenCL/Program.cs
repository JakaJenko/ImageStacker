using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Cloo;

namespace ClootilsNetCore
{
    class Program
    {
        public static long[] globalWorkSize;
        public static long[] localWorkSize;

        public static int Main(string[] args)
        {
            
            Console.WriteLine("Image path: ");
            string imagePath = Console.ReadLine();

            Console.WriteLine();
            Console.WriteLine("Main image name: ");
            string mainImageName = Console.ReadLine();

            Console.WriteLine();
            Console.WriteLine("First move distance (must be divisible by 2, else uses 64): ");
            string setMoveByString = Console.ReadLine();

            int setMoveBy;

            if (!Int32.TryParse(setMoveByString, out setMoveBy))
            {
                Console.WriteLine("First move distance set to 64");
                setMoveBy = 64;
            }
            else if (setMoveBy % 2 != 0)
            {
                Console.WriteLine("First move distance set to 64");
                setMoveBy = 64;
            }

            if (!imagePath.EndsWith("\\"))
                imagePath += "\\";

            Bitmap mainImageOriginal = new Bitmap(imagePath + mainImageName);
            Bitmap mainImage = mainImageOriginal; //ResizeImage(mainImageOriginal, mainImageOriginal.Width * 2, mainImageOriginal.Height * 2);

            BitmapData mainImageBitmapData = mainImage.LockBits(new Rectangle(0, 0, mainImage.Width, mainImage.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            
            int width = mainImage.Width;
            int height = mainImage.Height;

            Console.WriteLine(width);
            Console.WriteLine(height);

            Int32 mainImageStride = mainImageBitmapData.Stride;
            Byte[] mainImageData = new Byte[mainImageStride * height];
            Marshal.Copy(mainImageBitmapData.Scan0, mainImageData, 0, mainImageData.Length);

            Console.WriteLine(String.Join(" ", mainImageData.Where(x=>x<0).Take(100)));
            
            

            //var sw = new Stopwatch();
            //sw.Restart();

            // pick first platform
            ComputePlatform platform = Cloo.ComputePlatform.Platforms[0];
            Console.WriteLine(platform.Name);
            Console.WriteLine("Max work items in work group: " + platform.Devices[0].MaxWorkGroupSize);
            Console.WriteLine(platform.Devices[0].MaxWorkItemDimensions);
            Console.WriteLine(String.Join(" ", platform.Devices[0].MaxWorkItemSizes));

            // work sizes
            localWorkSize = new long[] { platform.Devices[0].MaxWorkGroupSize };

            long globalWorkSizeTmp = mainImageData.Length / 4;
            long multiplier = Convert.ToInt64(Math.Ceiling((float)globalWorkSizeTmp / localWorkSize[0]));

            Console.WriteLine(globalWorkSizeTmp);
            Console.WriteLine(localWorkSize[0] * multiplier);

            globalWorkSize = new long[] { localWorkSize[0] * multiplier };
            

            // create context with all gpu devices
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu,
                                                                new ComputeContextPropertyList(platform),
                                                                null,
                                                                IntPtr.Zero);

            // create a command queue with first gpu found
            ComputeCommandQueue queue = new ComputeCommandQueue(context,
                                                                context.Devices[0],
                                                                ComputeCommandQueueFlags.None);



            Console.WriteLine("Data len: " + mainImageData.Length);
            var mainImageBuffer = new ComputeBuffer<byte>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, mainImageData);


            var partialSums = new long[globalWorkSize[0] / localWorkSize[0]];

            var partialSumsBuffer = new ComputeBuffer<long>(context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, partialSums);
            GCHandle arrCHandlePartialSumsData = GCHandle.Alloc(partialSums, GCHandleType.Pinned);


            // create program with opencl source
            ComputeProgram program = new ComputeProgram(context, File.ReadAllText(Path.Combine(Directory.GetCurrentDirectory(), "kernelGetDifference.c")));

            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception ex)
            {
                String buildLog = program.GetBuildLog(context.Devices[0]);
                Console.WriteLine("\n********** Build Log **********\n" + buildLog + "\n*************************");
            }


            // load chosen kernel from program
            ComputeKernel kernel = program.CreateKernel("kernelGetDifference");


            kernel.SetValueArgument(0, width);
            kernel.SetValueArgument(1, height);

            kernel.SetMemoryArgument(4, mainImageBuffer);

            kernel.SetMemoryArgument(6, partialSumsBuffer);
            kernel.SetLocalArgument(7, localWorkSize[0] * 4); // * 4 -> int = 4 bytes


            //  file name, differnce, was moved, (moved by width, moved by height)
            var results = new List<Tuple<string, long, bool, Tuple<int, int>>>();

            string imgName;
            foreach (string imageFileName in Directory.GetFiles(imagePath).Where(s => (s.EndsWith(".png") || s.EndsWith(".jpg") && !s.EndsWith(mainImageName))))
            {
                imgName = imageFileName;
                Console.WriteLine(imgName);
               

                using (Bitmap moveImage = ResizeImage(new Bitmap(imgName), width, height))
                {
                    BitmapData moveImageBitmapData = moveImage.LockBits(new Rectangle(0, 0, moveImage.Width, moveImage.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                    Int32 moveImageStride = moveImageBitmapData.Stride;
                    Byte[] moveImageData = new Byte[moveImageStride * moveImage.Height];
                    Marshal.Copy(moveImageBitmapData.Scan0, moveImageData, 0, moveImageData.Length);

                    var moveImageBuffer = new ComputeBuffer<byte>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, moveImageData);

                    kernel.SetMemoryArgument(5, moveImageBuffer);


                    Tuple<int, int> moveWidthHeight = new Tuple<int, int>(0, 0);

                    long lastDifference;
                    long newDifference = long.MaxValue;

                    int prevMoveBy;
                    int moveBy = setMoveBy;

                    Tuple<string, long, bool, Tuple<int, int>> bestResult;

                    do
                    {
                        var partialResults = new List<Tuple<string, long, bool, Tuple<int, int>>>();
                        lastDifference = newDifference;

                        partialResults.Add(new Tuple<string, long, bool, Tuple<int, int>>(imgName, GetDifferenceBetweenImages(kernel, queue, partialSums, partialSumsBuffer, arrCHandlePartialSumsData, moveWidthHeight.Item1, moveWidthHeight.Item2), false, new Tuple<int, int>(moveWidthHeight.Item1, moveWidthHeight.Item2)));
                        partialResults.Add(new Tuple<string, long, bool, Tuple<int, int>>(imgName, GetDifferenceBetweenImages(kernel, queue, partialSums, partialSumsBuffer, arrCHandlePartialSumsData, moveWidthHeight.Item1 + moveBy, moveWidthHeight.Item2), true, new Tuple<int, int>(moveWidthHeight.Item1 + moveBy, moveWidthHeight.Item2)));
                        partialResults.Add(new Tuple<string, long, bool, Tuple<int, int>>(imgName, GetDifferenceBetweenImages(kernel, queue, partialSums, partialSumsBuffer, arrCHandlePartialSumsData, moveWidthHeight.Item1 - moveBy, moveWidthHeight.Item2), true, new Tuple<int, int>(moveWidthHeight.Item1 - moveBy, moveWidthHeight.Item2)));
                        partialResults.Add(new Tuple<string, long, bool, Tuple<int, int>>(imgName, GetDifferenceBetweenImages(kernel, queue, partialSums, partialSumsBuffer, arrCHandlePartialSumsData, moveWidthHeight.Item1, moveWidthHeight.Item2 + moveBy), true, new Tuple<int, int>(moveWidthHeight.Item1, moveWidthHeight.Item2 + moveBy)));
                        partialResults.Add(new Tuple<string, long, bool, Tuple<int, int>>(imgName, GetDifferenceBetweenImages(kernel, queue, partialSums, partialSumsBuffer, arrCHandlePartialSumsData, moveWidthHeight.Item1, moveWidthHeight.Item2 - moveBy), true, new Tuple<int, int>(moveWidthHeight.Item1, moveWidthHeight.Item2 - moveBy)));

                        var smallestDifference = partialResults.Aggregate((curMin, x) => curMin == null || curMin.Item2 > x.Item2 ? x : curMin);
                        bestResult = smallestDifference;

                        newDifference = smallestDifference.Item2;
                        moveWidthHeight = smallestDifference.Item4;

                        prevMoveBy = moveBy;

                        //if was not moved, reduce move by
                        if (!smallestDifference.Item3)
                            moveBy /= 2;

                    } while (lastDifference > newDifference || prevMoveBy > 1);

                    results.Add(bestResult);
                    Console.WriteLine(newDifference);
                }

                System.GC.Collect();
                System.GC.WaitForPendingFinalizers();
            }


            // Combine images
            program = new ComputeProgram(context, File.ReadAllText(Path.Combine(Directory.GetCurrentDirectory(), "kernelCombineImages.c")));

            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception ex)
            {
                String buildLog = program.GetBuildLog(context.Devices[0]);
                Console.WriteLine("\n********** Build Log **********\n" + buildLog + "\n*************************");
            }

            kernel = program.CreateKernel("kernelCombineImages");
            kernel.SetValueArgument(0, width);
            kernel.SetValueArgument(1, height);

            kernel.SetMemoryArgument(4, mainImageBuffer);

            for (int resultId = 0; resultId < results.Count; resultId++)
            {
                var result = results[resultId];

                using (Bitmap imageToAdd = ResizeImage(new Bitmap(result.Item1), width, height))
                {
                    Console.WriteLine("Final diff: " + result.Item2 + " " + result.Item4.Item1 + "|" + result.Item4.Item2);

                    BitmapData imageToAddBitmapData = imageToAdd.LockBits(new Rectangle(0, 0, imageToAdd.Width, imageToAdd.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                    Int32 imageToAddStride = imageToAddBitmapData.Stride;
                    Byte[] imageToAddData = new Byte[imageToAddStride * imageToAdd.Height];
                    Marshal.Copy(imageToAddBitmapData.Scan0, imageToAddData, 0, imageToAddData.Length);

                    var imageToAddImageBuffer = new ComputeBuffer<byte>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, imageToAddData);

                    kernel.SetValueArgument(2, result.Item4.Item1);
                    kernel.SetValueArgument(3, result.Item4.Item2);

                    kernel.SetMemoryArgument(5, imageToAddImageBuffer);
                    kernel.SetValueArgument(6, resultId + 2);
                    
                    Console.WriteLine("------------");
                    queue.Execute(kernel, null, globalWorkSize, localWorkSize, null);
                    Console.WriteLine("------------");

                    queue.Finish();
                }

            }

            Int32 combinedImageStride = mainImageBitmapData.Stride;
            Byte[] combinedImageData = new Byte[combinedImageStride * height];
            GCHandle arrCHandleCombinedImageData = GCHandle.Alloc(combinedImageData, GCHandleType.Pinned);
            queue.Read(mainImageBuffer, false, 0, combinedImageData.Length, arrCHandleCombinedImageData.AddrOfPinnedObject(), null);
            queue.Finish();

            //queue.Read(imgBuffer1, false, 0, dataImg1.Length, arrCHandleImg1Data.AddrOfPinnedObject(), events);
            //queue.Read(imgBuffer2, false, 0, dataImg2.Length, arrCHandleImg2Data.AddrOfPinnedObject(), events);


            GetDataPicture(width, height, combinedImageData).Save(Path.Combine(Directory.GetCurrentDirectory(),  "combined.png"));



            // wait for completion
            //queue.Finish();
            //sw.Stop();
            //Console.WriteLine($"{sw.ElapsedMilliseconds}ms");

            Console.WriteLine("DONE");
            Console.ReadLine();
            return 0;
        }

        public static long GetDifferenceBetweenImages(ComputeKernel kernel, ComputeCommandQueue queue, long[] partialSums, ComputeBuffer<long> partialSumsBuffer, GCHandle arrCHandlePartialSumsData, int moveWidth, int moveHeight)
        {
            kernel.SetValueArgument(2, moveWidth);
            kernel.SetValueArgument(3, moveHeight);

            queue.Execute(kernel, null, globalWorkSize, localWorkSize, null);
            queue.Finish();

            queue.Read(partialSumsBuffer, false, 0, partialSums.Length, arrCHandlePartialSumsData.AddrOfPinnedObject(), null);
            queue.Finish();

            return partialSums.Sum();
        }


        public static Bitmap GetDataPicture(int width, int height, byte[] data)
        {
            Console.WriteLine(width);
            Console.WriteLine(height);
            Bitmap img = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int arrayIndex = (h * width + w) * 4;

                    Color c = Color.FromArgb(
                       data[arrayIndex + 3],
                       data[arrayIndex + 2],
                       data[arrayIndex + 1],
                       data[arrayIndex]
                    );

                    img.SetPixel(w, h, c);
                }
            }

            return img;
        }

        public static Bitmap GetDataPicture(int width, int height, int[] data)
        {
            Console.WriteLine(width);
            Console.WriteLine(height);
            Bitmap img = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int arrayIndex = h * width + w;

                    Color c = Color.FromArgb(
                      255,
                      data[arrayIndex] % 255,
                      data[arrayIndex] % 255,
                      data[arrayIndex] % 255
                   );

                    if (data[arrayIndex] != 0)
                    {
                        c = Color.FromArgb(
                          255,
                          255,
                          255,
                          255
                       );
                    }

                   

                    img.SetPixel(w, h, c);
                }
            }

            return img;
        }

        /// <summary>
        /// From Stackoverflow (https://stackoverflow.com/questions/1922040/how-to-resize-an-image-c-sharp)
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <returns>The resized image.</returns>
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}