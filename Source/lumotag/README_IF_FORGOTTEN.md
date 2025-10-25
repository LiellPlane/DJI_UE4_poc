the system uses a server (although will run without), and expects to be on the lumotag wifi (see settings in setup)

get that running first

then start up the server in the typescript-server-app folder, use pnpm build:start:dashboard
look at device-tag-mapping.json for player information if you want to modify it
avatars go in packages/server/public/avatars - make sure they have the same captilisation!!

connect to the dashboard url to see live server status and expose some debug controls

now start the gun, it will get confused if the scambilight wifi is on (one device will anyway). It will try and find the lumotag wifi and at time
of writing, will try and download the newest repo from github. if you have problems, you can ssh in using password lumotag. Find it using putty for windows, or use angry IP scanner if you cannot connect


Look at the switches, it has two circuits. you need to put them off neutral to outer positions for both (so they move away from each other)

put ONE switch ONLY in the INNER position so it the battery can charge. You may be able to put both on depending on if the power has been upgraded

