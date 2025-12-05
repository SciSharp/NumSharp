# NumSharp使用指南及核心组件详解

本文档详细介绍了NumSharp数值计算库的核心组件NDArray、Slice和Shape的使用方法，包含创建、修改、访问和操作数组的具体示例。

## 目录

1. [NDArray基本操作](#ndarray基本操作)
2. [Slice切片操作](#slice切片操作)
3. [Shape形状管理](#shape形状管理)
4. [高级操作示例](#高级操作示例)

## NDArray基本操作

NDArray是NumSharp中的核心数据结构，代表多维同质数组。

### 创建NDArray

#### 从数组创建
```csharp
using NumSharp;

// 从C#数组创建
int[] arr1D = {1, 2, 3, 4, 5};
NDArray nd1 = new NDArray(arr1D);

double[,] arr2D = {{1.0, 2.0}, {3.0, 4.0}};
NDArray nd2 = new NDArray(arr2D);

// 使用np创建函数
NDArray zeros = np.zeros(3, 4);  // 3x4零矩阵
NDArray ones = np.ones(2, 3, 4); // 2x3x4全1矩阵
NDArray range = np.arange(10);    // [0, 1, 2, ..., 9]
NDArray linspace = np.linspace(0, 10, 5); // [0, 2.5, 5, 7.5, 10]
```

#### 指定数据类型创建
```csharp
// 指定数据类型和形状创建
NDArray intArr = new NDArray(typeof(int), new Shape(3, 4)); // 3x4整数数组
NDArray floatArr = new NDArray(NPTypeCode.Single, new Shape(2, 3)); // 2x3单精度浮点数组
```

### 访问和修改数据

#### 获取数组信息
```csharp
NDArray arr = np.arange(12).reshape(3, 4);

Console.WriteLine($"数据类型: {arr.dtype}");      // 数据类型: System.Int32
Console.WriteLine($"形状: [{string.Join(", ", arr.shape)}]"); // 形状: [3, 4]
Console.WriteLine($"维度数: {arr.ndim}");         // 维度数: 2
Console.WriteLine($"元素总数: {arr.size}");       // 元素总数: 12
Console.WriteLine($"数据大小: {arr.dtypesize}");   // 数据大小: 4 (字节)
```

#### 元素访问
```csharp
NDArray arr = np.array(new int[] {10, 20, 30, 40});

// 访问单个元素
int val = arr.GetInt32(2); // val = 30
Console.WriteLine($"第三元素: {val}");

// 二维数组访问
NDArray mat = np.array(new int[,] {{1, 2, 3}, {4, 5, 6}});
int val2d = mat.GetInt32(1, 2); // val2d = 6 (第二行第三列)
Console.WriteLine($"(1,2)位置值: {val2d}");

// 通用访问方法
var genericVal = mat.GetValue<int>(0, 1); // genericVal = 2
```

#### 元素修改
```csharp
NDArray arr = np.arange(5);

// 修改单个元素
arr.SetValue(99, 2); // 将索引2处的元素改为99
Console.WriteLine($"修改后数组: {arr}");

// 批量修改
arr.SetData(new int[] {100, 200}, 1, 3); // 修改索引1和3处的值

// 修改二维数组
NDArray mat = np.zeros(2, 3);
mat.SetInt32(42, 0, 1); // 将(0,1)位置的值设为42
Console.WriteLine($"修改后矩阵:\n{mat}");
```

#### 批量数据操作
```csharp
// 获取内部数据的引用
NDArray arr = np.array(new double[] {1.1, 2.2, 3.3});
ArraySlice<double> data = arr.Data<double>();
Console.WriteLine($"数据: [{string.Join(", ", data)}]");

// 批量设置数据
NDArray target = new NDArray(typeof(double), 3);
target.SetData(new double[] {5.5, 6.6, 7.7});
Console.WriteLine($"目标数组: {target}");
```

## Slice切片操作

Slice类提供了类似NumPy的切片功能，可以方便地对数组进行部分访问和修改。

### 基本切片操作
```csharp
NDArray arr = np.arange(10); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

// 使用Slice对象
NDArray slice1 = arr[new Slice(2, 7)];    // [2, 3, 4, 5, 6] - 从索引2到6
NDArray slice2 = arr[new Slice(null, 5)]; // [0, 1, 2, 3, 4] - 从开始到索引4
NDArray slice3 = arr[new Slice(3, null)]; // [3, 4, 5, 6, 7, 8, 9] - 从索引3到结束

Console.WriteLine($"原数组: {arr}");
Console.WriteLine($"切片[2:7]: {slice1}");
Console.WriteLine($"切片[:5]: {slice2}");
Console.WriteLine($"切片[3:]: {slice3}");
```

### 带步长的切片
```csharp
NDArray arr = np.arange(10); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

// 步长切片
NDArray stepSlice = arr[new Slice(0, null, 2)]; // [0, 2, 4, 6, 8] - 步长为2
NDArray reverseSlice = arr[new Slice(null, null, -1)]; // [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - 反转
NDArray reverseStep = arr[new Slice(null, null, -2)]; // [9, 7, 5, 3, 1] - 反转步长为2

Console.WriteLine($"步长为2的切片: {stepSlice}");
Console.WriteLine($"反向切片: {reverseSlice}");
Console.WriteLine($"反向步长为2的切片: {reverseStep}");
```

### 多维数组切片
```csharp
// 创建3x4数组
NDArray mat = np.arange(12).reshape(3, 4);
/*
mat = 
[[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]]
*/

// 行切片
NDArray rowSlice = mat[new Slice(1, 3), Slice.All]; // 第1到2行
Console.WriteLine($"第1-2行:\n{rowSlice}");

// 列切片
NDArray colSlice = mat[Slice.All, new Slice(0, 2)]; // 第0到1列
Console.WriteLine($"第0-1列:\n{colSlice}");

// 组合切片
NDArray subMat = mat[new Slice(0, 2), new Slice(1, 3)]; // 前2行的第1-2列
Console.WriteLine($"子矩阵:\n{subMat}");

// 单个索引（会减少一个维度）
NDArray row = mat[1, Slice.All]; // 第1行，结果是1维数组
Console.WriteLine($"第1行: {row}");

NDArray col = mat[Slice.All, 2]; // 第2列，结果是1维数组
Console.WriteLine($"第2列: {col}");
```

### 使用字符串切片表示法
```csharp
NDArray arr = np.arange(10);

// 使用字符串表示法
NDArray slice1 = arr["2:7"];    // 等同于 new Slice(2, 7)
NDArray slice2 = arr["::2"];    // 等同于 new Slice(null, null, 2)
NDArray slice3 = arr["::-1"];   // 等同于 new Slice(null, null, -1)

Console.WriteLine($"字符串表示法切片'2:7': {slice1}");
Console.WriteLine($"字符串表示法切片'::2': {slice2}");
Console.WriteLine($"字符串表示法切片'::-1': {slice3}");
```

### 切片修改操作
```csharp
NDArray arr = np.arange(10);
Console.WriteLine($"原始数组: {arr}");

// 修改切片会影响原数组（因为切片是视图）
NDArray slice = arr[new Slice(2, 5)];
slice.fill(99); // 将切片中的所有元素设置为99
Console.WriteLine($"修改后的原数组: {arr}"); // [0, 1, 99, 99, 99, 5, 6, 7, 8, 9]

// 二维数组切片修改
NDArray mat = np.zeros(3, 4);
mat[new Slice(1, 3), new Slice(1, 3)] = np.ones(2, 2) * 5;
Console.WriteLine($"修改后的矩阵:\n{mat}");
```

## Shape形状管理

Shape类管理数组的形状信息，包括维度、大小和内存布局。

### Shape基本操作
```csharp
// 创建Shape对象
Shape shape1 = new Shape(3, 4);        // 3x4形状
Shape shape2 = new Shape(2, 3, 4);     // 2x3x4形状
Shape shape3 = new Shape(new int[] {5, 6, 7}); // 从数组创建

Console.WriteLine($"shape1: {shape1}, 维度: {shape1.NDim}, 大小: {shape1.Size}");
Console.WriteLine($"shape2: {shape2}, 维度: {shape2.NDim}, 大小: {shape2.Size}");
Console.WriteLine($"shape3: {shape3}, 维度: {shape3.NDim}, 大小: {shape3.Size}");

// Shape的维度访问
Console.WriteLine($"shape1的第0维: {shape1[0]}"); // 3
Console.WriteLine($"shape1的第1维: {shape1[1]}"); // 4
```

### Shape转换操作
```csharp
NDArray arr = np.arange(12);

// 重塑形状
NDArray reshaped1 = arr.reshape(3, 4);    // 3x4矩阵
NDArray reshaped2 = arr.reshape(2, 6);    // 2x6矩阵
NDArray reshaped3 = arr.reshape(2, 2, 3); // 2x2x3三维数组

Console.WriteLine($"原数组形状: {arr.shape}");
Console.WriteLine($"重塑为3x4:\n{reshaped1}");
Console.WriteLine($"重塑为2x6:\n{reshaped2}");
Console.WriteLine($"重塑为2x2x3:\n{reshaped3}");

// 扁平化
NDArray flat = reshaped1.flatten(); // 转换为一维数组
Console.WriteLine($"扁平化后: {flat}");
```

### Shape的高级操作
```csharp
NDArray arr = np.arange(24).reshape(2, 3, 4);

// 获取子形状
var (subShape, offset) = arr.Shape.GetSubshape(1); // 获取第二"页"的形状和偏移
Console.WriteLine($"子形状: {subShape}, 偏移: {offset}");

// 获取坐标
int linearIndex = 10;
int[] coords = arr.Shape.GetCoordinates(linearIndex);
Console.WriteLine($"线性索引{linearIndex}对应的坐标: [{string.Join(", ", coords)}]");

// 获取偏移
int[] testCoords = {1, 2, 3};
int offsetFromCoords = arr.Shape.GetOffset(testCoords);
Console.WriteLine($"坐标[{string.Join(", ", testCoords)}]对应的偏移: {offsetFromCoords}");
```

## 高级操作示例

### 数学运算
```csharp
// 基本数学运算
NDArray a = np.array(new double[] {1, 2, 3, 4});
NDArray b = np.array(new double[] {5, 6, 7, 8});

NDArray sum = a + b;           // [6, 8, 10, 12]
NDArray product = a * b;       // [5, 12, 21, 32]
NDArray subtract = b - a;      // [4, 4, 4, 4]
NDArray divide = b / a;        // [5, 3, 2.33, 2]

Console.WriteLine($"a: {a}");
Console.WriteLine($"b: {b}");
Console.WriteLine($"a + b: {sum}");
Console.WriteLine($"a * b: {product}");

// 一元运算
NDArray sqrtA = np.sqrt(a);    // [1, 1.41, 1.73, 2]
NDArray squareA = np.square(a); // [1, 4, 9, 16]
NDArray absA = np.abs(a - 2.5); // [1.5, 0.5, 0.5, 1.5]

Console.WriteLine($"sqrt(a): {sqrtA}");
Console.WriteLine($"square(a): {squareA}");
Console.WriteLine($"abs(a - 2.5): {absA}");
```

### 统计操作
```csharp
NDArray arr = np.array(new double[,] {{1, 2, 3}, {4, 5, 6}});

// 基本统计
double sumAll = np.sum(arr).GetDouble();       // 21
double meanAll = np.mean(arr).GetDouble();     // 3.5
double maxAll = np.max(arr).GetDouble();       // 6
double minAll = np.min(arr).GetDouble();       // 1

Console.WriteLine($"总和: {sumAll}, 平均值: {meanAll}");
Console.WriteLine($"最大值: {maxAll}, 最小值: {minAll}");

// 按轴统计
NDArray sumAxis0 = np.sum(arr, axis: 0);  // [5, 7, 9] - 按列求和
NDArray sumAxis1 = np.sum(arr, axis: 1);  // [6, 15] - 按行求和

Console.WriteLine($"按列求和: {sumAxis0}");
Console.WriteLine($"按行求和: {sumAxis1}");

// 获取最值索引
int argMax = np.argmax(arr).GetInt32();   // 5 (最大值6的索引)
int argMin = np.argmin(arr).GetInt32();   // 0 (最小值1的索引)

Console.WriteLine($"最大值索引: {argMax}, 最小值索引: {argMin}");
```

### 线性代数操作
```csharp
// 矩阵乘法
NDArray matA = np.array(new double[,] {{1, 2}, {3, 4}});
NDArray matB = np.array(new double[,] {{5, 6}, {7, 8}});

NDArray matMul = np.dot(matA, matB);
Console.WriteLine($"矩阵A:\n{matA}");
Console.WriteLine($"矩阵B:\n{matB}");
Console.WriteLine($"A * B:\n{matMul}");

// 转置
NDArray transposed = matA.T; // 或者使用 np.transpose(matA)
Console.WriteLine($"A的转置:\n{transposed}");

// 其他操作
NDArray reshaped = np.arange(12).reshape(3, 4);
NDArray transposed2D = np.transpose(reshaped);
Console.WriteLine($"原矩阵:\n{reshaped}");
Console.WriteLine($"转置后:\n{transposed2D}");
```

### 数组操作
```csharp
// 数组合并
NDArray arr1 = np.array(new int[] {1, 2, 3});
NDArray arr2 = np.array(new int[] {4, 5, 6});

NDArray hStack = np.hstack(new NDArray[] {arr1, arr2}); // 水平拼接
NDArray vStack = np.vstack(new NDArray[] {arr1, arr2}); // 垂直拼接

Console.WriteLine($"原数组1: {arr1}");
Console.WriteLine($"原数组2: {arr2}");
Console.WriteLine($"水平拼接: {hStack}");
Console.WriteLine($"垂直拼接:\n{vStack}");

// 数组复制
NDArray original = np.array(new int[] {1, 2, 3});
NDArray cloned = original.Clone(); // 深拷贝
NDArray view = original.view();    // 视图（共享数据）

Console.WriteLine($"原数组: {original}");
Console.WriteLine($"拷贝: {cloned}");
Console.WriteLine($"视图: {view}");

// 类型转换
NDArray floatArr = original.astype(typeof(float));
Console.WriteLine($"转换为float: {floatArr}, 类型: {floatArr.dtype}");
```

### 条件操作
```csharp
NDArray arr = np.array(new int[] {-2, -1, 0, 1, 2});

// 条件选择
NDArray positive = np.where(arr > 0, arr, 0); // 大于0的保持，否则设为0
NDArray absValues = np.abs(arr);              // 绝对值

Console.WriteLine($"原数组: {arr}");
Console.WriteLine($"正数保持: {positive}");
Console.WriteLine($"绝对值: {absValues}");

// 布尔索引（通过条件获取索引）
NDArray condition = arr > 0;
NDArray positiveOnly = arr[condition]; // 只获取正数
Console.WriteLine($"正数元素: {positiveOnly}");
```

### 排序和搜索
```csharp
NDArray arr = np.array(new int[] {3, 1, 4, 1, 5, 9, 2, 6, 5});

// 排序
NDArray sorted = np.sort(arr);
NDArray sortedIndices = np.argsort(arr); // 返回排序后的索引

Console.WriteLine($"原数组: {arr}");
Console.WriteLine($"排序后: {sorted}");
Console.WriteLine($"排序索引: {sortedIndices}");

// 唯一值
NDArray unique = np.unique(arr);
Console.WriteLine($"唯一值: {unique}");
```

### 广播操作
```csharp
// NumSharp支持广播操作
NDArray mat = np.array(new double[,] {{1, 2, 3}, {4, 5, 6}});
NDArray vec = np.array(new double[] {10, 20, 30});

// 向量与矩阵的广播加法
NDArray result = mat + vec; // 每行都加上向量vec
Console.WriteLine($"矩阵:\n{mat}");
Console.WriteLine($"向量: {vec}");
Console.WriteLine($"广播加法结果:\n{result}");

// 标量操作（自动广播到所有元素）
NDArray scaled = mat * 2.0;
Console.WriteLine($"矩阵乘以2:\n{scaled}");
```