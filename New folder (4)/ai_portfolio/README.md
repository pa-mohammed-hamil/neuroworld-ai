# AI Engineering Portfolio

Responsive, dependency-free portfolio website for PA Mohammed Hamil.

## Run locally
Open `index.html` in a browser, or run `python -m http.server 8000` from this folder and visit `http://localhost:8000`.

## Add actual working project demos
1. Create `assets/demos/` and place a project video there, e.g. `lexora.mp4`.
2. In `index.html`, find `const projects=[` and change the matching record from `demo:null` to `demo:'assets/demos/lexora.mp4'`.
3. Add the real project repository URL by changing `repo:null` to `repo:'https://github.com/.../...'`.
4. Update project statuses and summaries to reflect actual implementation. Never claim unimplemented features are complete.
5. For live deployed applications, add a `live` URL field and link it from project cards if desired.

## Deploy
Upload the folder to GitHub and deploy as a static website using Vercel, Netlify, or GitHub Pages. No build step required.

## Customize
Update contact links, actual email, verified achievements and role descriptions in `index.html`. The template deliberately omits an email address and fabricated demo links.

## Full project portfolio
All 13 current project concepts are listed in index.html. Project demo URLs and repository URLs remain null until real links are available. Edit the `projects` array to update statuses, demos, and repositories.
