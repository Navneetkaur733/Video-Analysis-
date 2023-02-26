from getmac import get_mac_address
import wsdiscovery as wsd
import onvif
import json
import re
import os


def fetch_onvif_devices():
    devices = []
    wsd_client = wsd.discovery.ThreadedWSDiscovery()
    print('WS-Discovery Initiated ...')
    scope = wsd.Scope("onvif://www.onvif.org/Profile")
    wsd_client.start()
    print('WS-Started Searching ...')
    services = wsd_client.searchServices(scopes=[scope])
    for service in services: 
        x_addresses = service.getXAddrs()
        if not len(x_addresses): continue
        ipaddress = re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', str(x_addresses[0]))[0]
        ports = re.findall(r':\d+',str(x_addresses[0]))
        port = ports[0][1::] if len(ports) else 80
        print('Device Found at ',ipaddress, port)
        device_scopes =[str(sc).split(r'/')[-2:] for sc in service.getScopes()]
        device = dict(device_scopes)
        device.update({
            'ip': ipaddress, 
            'port':port, 
            'x_addr':str(x_addresses[0]), 
            'mac' : get_mac_address(ip=ipaddress)
            })
        devices.append(device)
    wsd_client.stop()
    return devices
 
def fetch_rtsp_uri(onvif_device):
    print(onvif_device)
    if 'uri' in onvif_device: return onvif_device.get('uri');
    username, password = onvif_device.get('username'), onvif_device.get('password')
    if username is None:
        username = input("Enter Username : ")
        onvif_device['username'] = username
    if password is None:
        password = input("Enter Password : ")
        onvif_device['password'] = password
    uri = f"rtsp://{username}:{password}@{onvif_device.get('ip')}:554/"
    cam = onvif.ONVIFCamera(onvif_device.get('ip'), onvif_device.get('port'), username, password, r".\env\Lib\site-packages\wsdl")
    media = cam.create_media_service()
    for profile in media.GetProfiles():
        stream_uri =  media.GetStreamUri({'StreamSetup':{'Stream':'RTP-Unicast','Transport':'UDP'},'ProfileToken':profile.token})
        uri = stream_uri.Uri
        uri = f'{uri[:7]}{username}:{password}@{uri[7:]}'
        onvif_device['uri'] = uri
        return uri
    return uri

def move(ptz, token, pos):       
     move_type = ptz.create_type('ContinuousMove')
     move_type.ProfileToken = token
     move_type.Velocity = ptz.GetStatus({'ProfileToken': token}).Position
     move_type.Velocity.PanTilt.x = pos[0]
     move_type.Velocity.PanTilt.y = pos[1]
     ptz.ContinuousMove(move_type)
     ptz.Stop({'ProfileToken': move_type.ProfileToken})

if __name__ == "__main__":
    file_path = 'res/onvif_devices.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            onvif_devices = json.load(file)
    else:
        print('Looking for new devices')
        onvif_devices = fetch_onvif_devices()

    for onvif_device in onvif_devices:
        uri = fetch_rtsp_uri(onvif_device)
        print("RTSP : ",uri)
    
    print('Saving Devices ...')
    with open(file_path, 'w') as file:
        file.write(json.dumps(onvif_devices, indent=4))

