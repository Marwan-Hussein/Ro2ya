using CommunityToolkit.Mvvm.Input;
using Ro2ya.Models;

namespace Ro2ya.PageModels
{
    public interface IProjectTaskPageModel
    {
        IAsyncRelayCommand<ProjectTask> NavigateToTaskCommand { get; }
        bool IsBusy { get; }
    }
}