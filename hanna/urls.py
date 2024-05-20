from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),

    path('chat/', views.chat_stream, name='chat'),

    path('create-collection/', views.create_collection),
    path('add-object/text/', views.add_vectors),
    path('add-object/file/', views.upload_file),

    path('remove-object/', views.remove_object),
    path('remove-objects/entity/', views.remove_objects_entity),
    path('remove-objects/uuid/', views.remove_objects_uuid),
    path('remove-collection/', views.remove_collection),

    path('get-object/', views.get_object),
    path('get-objects/entity/', views.get_objects_entity),
    path('get-objects/uuid/', views.get_objects_uuid),
    path('get-collection/', views.get_collection),

    # ---------- Master Vectors ----------
    path('create-master-collection/', views.create_master_collection),
    path('add-master-object/text/', views.add_master_vectors),
    path('add-master-object/file/', views.upload_master_file),

    path('remove-master-object/', views.remove_master_object),
    path('remove-master-objects/filename/', views.remove_master_objects_file),
    path('remove-master-objects/uuid/', views.remove_master_objects_uuid),
    path('remove-master-collection/', views.remove_master_collection),

    path('get-master-object/', views.get_master_object),
    path('get-master-objects/filename/', views.get_master_objects_filename),
    path('get-master-objects/uuid/', views.get_master_objects_uuid),
    path('get-master-objects/type/', views.get_master_objects_type),
    path('get-master-collection/', views.get_master_collection),

    # ---- CALL THIS WHEN LIMIT EXCEEDS!!!!!
    path('destroy/', views.destroy_all),

]
