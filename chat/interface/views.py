from django.shortcuts import render


def main_view(request):
    '''
    Main view which launch and handle the chatbot view
    Args:
        request <obj>: django request object
    '''
    return render(request, 'index.html', {})
