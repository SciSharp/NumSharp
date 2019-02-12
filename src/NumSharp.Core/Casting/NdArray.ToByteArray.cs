using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public byte[] ToByteArray()
        {
            var nd = ravel();
            byte[] bytes = new byte[size * dtypesize];
            byte[] cache = new byte[dtypesize];

            switch (dtype.Name)
            {
                case "Int16":
                    for (int i = 0; i < size; i++)
                    {
                        cache = BitConverter.GetBytes(nd.Data<short>(i));
                        for (int j = 0; j < dtypesize; j++)
                            bytes[i * dtypesize + j] = cache[j];
                    }
                    break;
                case "Int32":
                    for (int i = 0; i < size; i++)
                    {
                        cache = BitConverter.GetBytes(nd.Data<int>(i));
                        for (int j = 0; j < dtypesize; j++)
                            bytes[i * dtypesize + j] = cache[j];
                    }
                    break;
                case "Single":
                    for (int i = 0; i < size; i++)
                    {
                        cache = BitConverter.GetBytes(nd.Data<float>(i));
                        for (int j = 0; j < dtypesize; j++)
                            bytes[i * dtypesize + j] = cache[j];
                    }
                    break;
                case "Double":
                    for (int i = 0; i < size; i++)
                    {
                        cache = BitConverter.GetBytes(nd.Data<double>(i));
                        for (int j = 0; j < dtypesize; j++)
                            bytes[i * dtypesize + j] = cache[j];
                    }
                    break;
                default:
                    throw new NotImplementedException("NDArray ToByteArray() not implemented");
            }

            return bytes;
        }
    }
}
