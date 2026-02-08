# #424: The type or namespace name 'NumSharp' could not be found (are you missing a using directive or an assembly reference?) [Assembly-CSharp]csharp(CS0246)

- **URL:** https://github.com/SciSharp/NumSharp/issues/424
- **State:** OPEN
- **Author:** @rcffc
- **Created:** 2020-10-15T15:47:16Z
- **Updated:** 2020-10-19T21:52:32Z

## Description

I am using VSCode and have tried installing Numsharp using NuGet Gallery and Nuget Package Manager.

But still I am getting this error in my Unity project:
`The type or namespace name 'NumSharp' could not be found (are you missing a using directive or an assembly reference?) [Assembly-CSharp]csharp(CS0246)`

Any cues?

## Comments

### Comment 1 by @rcffc (2020-10-15T15:48:24Z)

It is also included in Assembly-CSharp.csproj:
    `<PackageReference Include="NumSharp" Version="0.20.5" />`

### Comment 2 by @ (2020-10-19T21:52:31Z)

Hello @rcffc, 

1) Have you tried to restart VSCode?

2) If your project is using .NET Core, the CLI tool allows you to easily install NuGet packages from VSCode.

   `dotnet add package NumSharp`

   After the command completes, look at the project file (*.csproj) to make sure the package was installed.

