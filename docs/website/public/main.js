// NumSharp DocFX Template Customization
// https://dotnet.github.io/docfx/docs/template.html

export default {
  // Icon links displayed in the navbar (top-right)
  iconLinks: [
    {
      icon: 'github',
      href: 'https://github.com/SciSharp/NumSharp',
      title: 'GitHub'
    },
    {
      icon: 'box-seam',
      href: 'https://www.nuget.org/packages/NumSharp',
      title: 'NuGet'
    }
  ],

  // Default theme: 'light', 'dark', or 'auto' (system preference)
  // defaultTheme: 'auto',

  // Startup script (runs when page loads)
  start: () => {
    // Add any custom initialization here
    // console.log('NumSharp docs loaded');
  },

  // Customize syntax highlighting (highlight.js)
  // configureHljs: (hljs) => {
  //   // Register additional languages or customize
  // }
}
