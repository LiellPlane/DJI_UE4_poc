#!/usr/bin/env python3
"""
WebSocket Connection Health Monitor

Monitors websocket connections for device-side issues:
- Connection drops/reconnections
- Message flow interruptions
- Network timeouts
- Buffer overflows
- Client behavior patterns
"""
import asyncio
import websockets
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading


class ConnectionMonitor:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.is_running = False
        self.clients: Dict[str, Dict] = {}
        self.connection_events: List[Dict] = []
        self.image_flow_events: List[Dict] = []
        
    def _log_event(self, event_type: str, client_id: str, details: Dict):
        """Log connection and flow events with timestamps"""
        event = {
            'timestamp': time.time(),
            'datetime': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'type': event_type,
            'client_id': client_id,
            'details': details
        }
        
        if event_type in ['connect', 'disconnect', 'reconnect']:
            self.connection_events.append(event)
        elif event_type in ['image_received', 'flow_stop', 'flow_resume']:
            self.image_flow_events.append(event)
        
        # Keep only last 100 events
        if len(self.connection_events) > 100:
            self.connection_events.pop(0)
        if len(self.image_flow_events) > 100:
            self.image_flow_events.pop(0)
            
        # Print real-time events
        print(f"[{event['datetime']}] {event_type.upper()}: {client_id} - {details}")
    
    async def handle_client(self, websocket, path=None):
        """Monitor client connections and message flow"""
        try:
            client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except:
            client_addr = "unknown"
            
        client_id = f"client_{len(self.clients)}_{client_addr}"
        
        # Initialize client tracking
        self.clients[client_id] = {
            'websocket': websocket,
            'address': client_addr,
            'connected_at': time.time(),
            'last_message': None,
            'message_count': 0,
            'image_count': 0,
            'last_image_time': None,
            'flow_active': True,
            'disconnected_at': None
        }
        
        self._log_event('connect', client_id, {
            'address': client_addr,
            'total_clients': len(self.clients)
        })
        
        try:
            last_activity = time.time()
            
            async for message in websocket:
                current_time = time.time()
                client_info = self.clients[client_id]
                client_info['last_message'] = current_time
                client_info['message_count'] += 1
                
                # Check for flow interruption (>5 seconds since last message)
                if current_time - last_activity > 5.0 and client_info['flow_active']:
                    self._log_event('flow_resume', client_id, {
                        'gap_seconds': current_time - last_activity,
                        'message_count': client_info['message_count']
                    })
                    client_info['flow_active'] = True
                
                last_activity = current_time
                
                try:
                    data = json.loads(message)
                    
                    # Track image messages specifically
                    if data.get('type') == 'image_upload' or 'image_data' in data:
                        client_info['image_count'] += 1
                        client_info['last_image_time'] = current_time
                        
                        image_id = data.get('image_id', 'unknown')
                        
                        # Log every 10th image to avoid spam
                        if client_info['image_count'] % 10 == 0:
                            self._log_event('image_received', client_id, {
                                'image_id': image_id,
                                'total_images': client_info['image_count'],
                                'rate_per_sec': client_info['image_count'] / (current_time - client_info['connected_at'])
                            })
                    
                except json.JSONDecodeError:
                    self._log_event('invalid_json', client_id, {
                        'message_length': len(message)
                    })
                except Exception as e:
                    self._log_event('message_error', client_id, {
                        'error': str(e)
                    })
                
        except websockets.exceptions.ConnectionClosed as e:
            self._log_event('disconnect', client_id, {
                'reason': 'connection_closed',
                'code': getattr(e, 'code', None),
                'duration': time.time() - self.clients[client_id]['connected_at'],
                'messages_received': self.clients[client_id]['message_count'],
                'images_received': self.clients[client_id]['image_count']
            })
        except Exception as e:
            self._log_event('disconnect', client_id, {
                'reason': 'error',
                'error': str(e),
                'duration': time.time() - self.clients[client_id]['connected_at']
            })
        finally:
            # Mark as disconnected but keep in history
            if client_id in self.clients:
                self.clients[client_id]['disconnected_at'] = time.time()
    
    async def _monitor_flow(self):
        """Monitor for flow interruptions"""
        while self.is_running:
            current_time = time.time()
            
            for client_id, client_info in self.clients.items():
                if client_info.get('disconnected_at'):
                    continue  # Skip disconnected clients
                    
                last_msg = client_info.get('last_message')
                if last_msg and current_time - last_msg > 10.0:  # No message for 10+ seconds
                    if client_info['flow_active']:
                        self._log_event('flow_stop', client_id, {
                            'seconds_since_last': current_time - last_msg,
                            'total_messages': client_info['message_count'],
                            'total_images': client_info['image_count']
                        })
                        client_info['flow_active'] = False
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
    
    async def _status_reporter(self):
        """Periodic status reports"""
        while self.is_running:
            await asyncio.sleep(30.0)  # Report every 30 seconds
            
            active_clients = [c for c in self.clients.values() if not c.get('disconnected_at')]
            total_images = sum(c['image_count'] for c in self.clients.values())
            
            print(f"\n📊 CONNECTION HEALTH REPORT [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"   Active clients: {len(active_clients)}")
            print(f"   Total clients seen: {len(self.clients)}")
            print(f"   Total images received: {total_images}")
            
            if active_clients:
                for client_id, client_info in self.clients.items():
                    if client_info.get('disconnected_at'):
                        continue
                    
                    duration = time.time() - client_info['connected_at']
                    last_msg = client_info.get('last_message', client_info['connected_at'])
                    silence = time.time() - last_msg
                    
                    status = "🟢 ACTIVE" if silence < 10 else "🔴 SILENT"
                    print(f"   {client_id}: {status} - {client_info['image_count']} images, "
                          f"{duration:.1f}s connected, {silence:.1f}s since last message")
            
            # Show recent events
            recent_events = [e for e in self.connection_events[-5:] if time.time() - e['timestamp'] < 300]
            if recent_events:
                print(f"   Recent events:")
                for event in recent_events:
                    print(f"     [{event['datetime']}] {event['type']}: {event['client_id']}")
            
            print()
    
    async def start_async(self):
        """Start the monitoring server"""
        print(f"🔍 ConnectionMonitor: Starting on {self.host}:{self.port}")
        print(f"📊 Monitoring: connection drops, flow interruptions, client behavior")
        print(f"🎯 Purpose: Diagnose device-side connection issues")
        print()
        
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            self.is_running = True
            
            print(f"✅ Monitor running on ws://{self.host}:{self.port}")
            print(f"🔗 Point your device to this URL to monitor its behavior")
            print()
            
            # Start background monitoring tasks
            monitor_task = asyncio.create_task(self._monitor_flow())
            status_task = asyncio.create_task(self._status_reporter())
            
            await self.server.wait_closed()
            
        except Exception as e:
            print(f"❌ Monitor failed to start: {e}")
            raise
    
    def get_diagnostics(self):
        """Get diagnostic information"""
        return {
            'clients': self.clients,
            'connection_events': self.connection_events,
            'image_flow_events': self.image_flow_events
        }


async def main():
    monitor = ConnectionMonitor()
    
    try:
        await monitor.start_async()
    except KeyboardInterrupt:
        print("\n🛑 Monitor shutting down...")
        if monitor.server:
            monitor.server.close()
            await monitor.server.wait_closed()
        
        # Print final diagnostics
        print("\n📋 FINAL DIAGNOSTICS:")
        diagnostics = monitor.get_diagnostics()
        
        print(f"Total clients: {len(diagnostics['clients'])}")
        print(f"Connection events: {len(diagnostics['connection_events'])}")
        print(f"Flow events: {len(diagnostics['image_flow_events'])}")
        
        # Show patterns
        disconnects = [e for e in diagnostics['connection_events'] if e['type'] == 'disconnect']
        if disconnects:
            print(f"\nDisconnection patterns:")
            for event in disconnects[-5:]:
                details = event['details']
                print(f"  {event['datetime']}: {event['client_id']} - "
                      f"reason: {details.get('reason')}, "
                      f"duration: {details.get('duration', 0):.1f}s, "
                      f"images: {details.get('images_received', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
