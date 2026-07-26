using Microsoft.Extensions.DependencyInjection;

namespace Ro2ya
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();
            UserAppTheme = AppTheme.Dark;
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            var window = new Window(new AppShell())
            {
                Title = "Ro2ya",
                Width = 1380,
                Height = 900
            };
            return window;
        }
    }
}