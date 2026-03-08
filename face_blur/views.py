from django.http import HttpResponse
from django.template import loader

def sample(request):
  template = loader.get_template('sample.html')
  return HttpResponse(template.render())