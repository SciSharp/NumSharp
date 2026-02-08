# #423: "System.NotImplementedException: '' --> someArray = np.frombuffer(byteBuffer.ToArray<byte>(), np.uint32);

- **URL:** https://github.com/SciSharp/NumSharp/issues/423
- **State:** OPEN
- **Author:** @mehmetcanbalci-Notrino
- **Created:** 2020-10-09T16:10:25Z
- **Updated:** 2020-10-18T22:51:51Z

## Description

Hello, 
i got  this execption "System.NotImplementedException: ''" when i use this code ; 
someArray = np.frombuffer(byteBuffer.ToArray<byte>(), np.uint32);

if I will use np.int32, it is working as expected.

## Comments

### Comment 1 by @ (2020-10-18T22:51:51Z)

Hello @mehmetcanbalci-Notrino ,

You got an exception because data type "uint32" is not yet being implemented, only "int32" and "byte". 

  ```c#
        public static NDArray frombuffer(byte[] bytes, Type dtype)
        {

            //TODO! all types
            if (dtype.Name == "Int32")
            {
                var size = bytes.Length / InfoOf<int>.Size;
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt32(bytes, index * InfoOf<int>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == "Byte")
            {
                var size = bytes.Length / InfoOf<byte>.Size;
                var ints = bytes;
                return new NDArray(bytes);
            }

            throw new NotImplementedException("");
        }
```
